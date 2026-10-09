import cv2

for idx in [0, 1, 2, 3]:
  print(f'Testing /dev/video{idx}...')
  cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
  cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

  if cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
      print(f'✅ /dev/video{idx} is WORKING! Frame shape: {frame.shape}')
      cap.release()
      break
    else:
      print(f'❌ /dev/video{idx} opened but returned no frame.')
    cap.release()
  else:
    print(f'❌ /dev/video{idx} could not be opened.')