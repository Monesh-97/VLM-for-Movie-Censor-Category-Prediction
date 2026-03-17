import cv2
import numpy as np

print("Generating sample.mp4...")
out = cv2.VideoWriter('sample.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 2.0, (640, 480))
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 128, 128), (255, 255, 255)
]

for i in range(16):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = colors[i % len(colors)]
    cv2.putText(frame, f"Sample Scene {i+1}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    out.write(frame)
    
out.release()
print("Created sample.mp4 (8 seconds, 16 frames).")
