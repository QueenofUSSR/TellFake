"""Local single-image demonstration of the frozen P3 CLIP probe."""
import argparse
import hashlib
import io
import json
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image, ImageOps

from reproduction.proposal import models

HERE = Path(__file__).resolve().parent
MAX_BYTES = 20 * 1024**2


class Detector:
    def __init__(self, weights=None):
        config = json.loads((HERE.parent / 'proposal/P3_FROZEN.json').read_text())
        method = 'frozen_L14_resize'
        spec = next(s for s in config['models'] if s['method'] == method and s['seed'] == 42)
        classifier_path = Path(spec['checkpoint']) if weights is None else weights / 'frozen_L14_resize_seed42.pkl'
        raw = classifier_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec['checkpoint_sha256']:
            raise ValueError('Classifier checksum mismatch')
        self.classifier = pickle.loads(raw)
        self.threshold = spec['thresholds']['p05']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if weights is None:
            encoder, meta = models.build(method)
        else:
            from transformers import CLIPVisionModelWithProjection
            encoder = CLIPVisionModelWithProjection.from_pretrained(weights, local_files_only=True)
            meta = {'kind': 'features'}
        self.encoder = encoder.to(self.device).eval()
        self.transform = models.Transform(method, False, meta)
        self.lock = threading.Lock()

    @torch.inference_mode()
    def classify(self, data):
        with Image.open(io.BytesIO(data), formats=('JPEG', 'PNG')) as image:
            if image.width * image.height > 32_000_000:
                raise ValueError('Image exceeds 32 megapixels')
            image.load()
            pixels = self.transform(ImageOps.exif_transpose(image).convert('RGB'), 42)
        with self.lock:
            embeds = self.encoder(pixel_values=pixels.unsqueeze(0).to(self.device)).image_embeds
            features = torch.nn.functional.normalize(embeds, dim=-1).float().cpu().numpy()
            score = float(self.classifier.predict_proba(features)[0, 1])
        return dict(ai_probability=score, threshold=self.threshold,
                    label='AI-Generated' if score >= self.threshold else 'Real')


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, content_type, body):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def json(self, code, data):
        self.reply(code, 'application/json', json.dumps(data).encode())

    def do_GET(self):
        if self.path == '/':
            self.reply(200, 'text/html; charset=utf-8', (HERE / 'index.html').read_bytes())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != '/predict':
            return self.send_error(404)
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send_error(403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            size = 0
        if not 0 < size <= MAX_BYTES:
            return self.json(413, {'error': '请选择不超过 20 MiB 的 JPEG 或 PNG 文件。'})
        try:
            result = self.server.detector.classify(self.rfile.read(size))
        except (OSError, ValueError, Image.DecompressionBombError, SyntaxError):
            return self.json(400, {'error': '无法读取图片。请使用完整的 JPEG 或 PNG，且不超过 3200 万像素。'})
        self.json(200, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--weights', type=Path, help='Directory containing verified release weights')
    args = parser.parse_args()
    torch.set_num_threads(4)
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        server.detector = Detector(args.weights)
        print(f'Ready at http://127.0.0.1:{args.port} ({server.detector.device})', flush=True)
        server.serve_forever()
