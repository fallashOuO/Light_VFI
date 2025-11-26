import cv2
import base64
import torch
from flask import Flask, Response

# 初始化模型（和 interpolate_light_vfi.py 一樣）
from light_vfi_model import LightVFIUNet2D
model = LightVFIUNet2D(in_frames=4)
model.load_state_dict(torch.load("weights_light_base/light_vfi_base_final.pth", map_location="cpu"))
model.eval()

app = Flask(__name__)

def gen():
    cap = cv2.VideoCapture(0)  # 攝影機(或你可以換成影片)
    while True:
        ret, frame = cap.read()
        if not ret: break

        # 這裡放 LightVFI 的插值邏輯
        # TODO：你要我可以直接幫你塞進去

        _, jpg = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(jpg).decode('utf-8')
        yield (f"--frame\r\nContent-Type: image/jpeg\r\n\r\n{jpg.tobytes()}\r\n")

@app.route('/video')
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host="0.0.0.0", port=7860)
