import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
import cv2
import numpy as np
import imutils 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--pipeline_config_path', required=True,
  help='path to pipeline.config')
ap.add_argument('--tflite_model_path', required=True,
  help='path to tflite model')
ap.add_argument('--frame_size', type=int, default=None,
  help='frame size of detection: if not given searches through config file')
ap.add_argument('--diplay_bbox', default=True,
  help='whether or not to display bounding box, scores, and class on frame')
ap.add_argument('--display_frame', default=True,
  help='whether or not to display frame')
ap.add_argument('--threshold', default=0.5,
  help='minimum score for which to detect an object')
args = ap.parse_args()

def detect(detection_model, interpreter, input_tensor, session):
  """Run detection on an input image.

  Args:
    interpreter: tf.lite.Interpreter
    input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.

  Returns:
    A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
      and `detection_scores`).
  """
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # We use the original model for pre-processing, since the TFLite model doesn't
  # include pre-processing.
  preprocessed_image, _ = detection_model.preprocess(input_tensor)
  interpreter.set_tensor(input_details[0]['index'], preprocessed_image.eval(session=session))

  interpreter.invoke()

  boxes = interpreter.get_tensor(output_details[0]['index'])
  classes = interpreter.get_tensor(output_details[1]['index'])
  scores = interpreter.get_tensor(output_details[2]['index'])
  return boxes, classes, scores



if __name__ == "__main__":
  sess = tf.Session()
  configs = config_util.get_configs_from_pipeline_file(args.pipeline_config_path)
  model_config = configs['model']

  FRAME_SIZE = args.frame_size
  if args.frame_size is None:
    FRAME_SIZE = model_config.ssd.image_resizer.fixed_shape_resizer.height

  detection_model = model_builder.build(
      model_config=model_config, is_training=True)

  interpreter = tf.lite.Interpreter(model_path=args.tflite_model_path)
  interpreter.allocate_tensors()

  cap = cv2.VideoCapture(0)

  while cap.isOpened():

    ret, frame = cap.read() 
    if not ret:
      break

    else:
      frame = imutils.resize(frame, width=FRAME_SIZE)
      boxes, classes, scores = detect(detection_model, interpreter, tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.float32), sess)

      if (args.diplay_bbox & args.display_frame):
        for box, cl, score in zip(boxes[0], classes[0], scores[0]):
          if (score >= args.threshold):
            box *= FRAME_SIZE
            cv2.rectangle(frame, (box[3], box[2]), (box[1], box[0]), (0,255,0), 1) # draw bbox: in cv2 we want upper left corner as start
            # and bottom right as end thus but tensorflow gives array as [y_min, x_min, y_max, x_max]

            cv2.putText(frame, '{}'.format(cl), (box[3], box[0]), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

      if args.display_frame:
        cv2.imshow("frame", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cv2.destroyAllWindows()
