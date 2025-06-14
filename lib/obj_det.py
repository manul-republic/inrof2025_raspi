import cv2
import onnxruntime
import numpy as np
from multiprocessing import Process, Queue
import uuid

class ObjectDetector:
    def __init__(self, model_path="masters.onnx"):
        super().__init__()
        self.input_size = (416, 416)
        self.num_classes = 6
        self.CLASSES = (
            "red-ball",
            "blue-ball",
            "yellow-ball",
            "yellow-can",
            "pyramid",
            "bottle",
        )
        self._COLORS = np.array(
            [
                0.000, 0.447, 0.741,
                0.850, 0.325, 0.098,
                0.929, 0.694, 0.125,
                0.494, 0.184, 0.556,
                0.466, 0.674, 0.188,
                0.301, 0.745, 0.933,
                0.635, 0.078, 0.184,
            ]
        ).astype(np.float32).reshape(-1, 3)

        self.model_path = model_path
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.process = Process(target=self._worker_loop, args=(model_path, self.task_queue, self.result_queue), daemon=True)
        self.process.start()
        self._pending_tasks = {}
    
    def __del__(self):
        self.shutdown()
    
    def shutdown(self):
        """終了処理：プロセスを停止"""
        try:
            if self.process.is_alive():
                self.task_queue.put("STOP")
                self.process.join(timeout=1)
                if self.process.is_alive() and True:
                    print("[WARN] process terminated")
                    self.process.terminate()
                    self.process.join()
        except Exception as e:
            print(f"[ERROR] exception occured during terminating: {e}")
        finally:
            try:
                self.task_queue.close()
                self.result_queue.close()
                self.task_queue.join_thread()
                self.result_queue.join_thread()
            except Exception as e:
                print(f"[ERROR] exception occured during releasing queue: {e}")

    def _worker_loop(self, model_path, task_queue, result_queue):
        self.ort_session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        while True:
            task = task_queue.get()
            if task == "STOP":
                break
            task_id, input_image = task
            result = self._inference(input_image)
            result_queue.put((task_id, result[0]))
    
    def put(self, input_array):
        """タスクIDを発行して推論リクエストを送信"""
        task_id = str(uuid.uuid4())
        self._pending_tasks[task_id] = True
        self.task_queue.put((task_id, input_array))
        return task_id
    
    def get(self, task_id, timeout=None):
        """結果を待機して受け取る（同期）"""
        while True:
            result_task_id, result_data = self.result_queue.get(timeout=timeout)
            if result_task_id == task_id:
                self._pending_tasks.pop(task_id, None)
                return result_data
            else:
                self.result_queue.put((result_task_id, result_data)) # 他タスクなら一時保存（必要に応じて）

    def predict(self, input):
        id = self.put(input)
        return self.get(id, timeout=3)

    def _inference(self, input):
        img, r = self._preprocess(input)
        onnx_input = {self.ort_session.get_inputs()[0].name: img}
        onnx_outputs = self.ort_session.run(None, onnx_input)[0]
        return self._postprocess(onnx_outputs, r)
    
    def _preprocess(self, img, swap=(2, 0, 1)): # BGR->RGB
        if len(img.shape) == 3:
            padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.input_size, dtype=np.uint8) * 114

        r = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_AREA,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.expand_dims(padded_img, axis=0) #set dim to (1,3,H,W)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    
    def _postprocess(self, prediction, scale, num_classes=None, conf_thre=0.2, nms_thre=0.45, class_agnostic=False):
        # Use self.num_classes if not provided
        if num_classes is None:
            num_classes = self.num_classes

        # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
        box_corner = np.zeros_like(prediction)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):

            if image_pred.shape[0] == 0:
                continue

            # Get score and class with highest confidence
            class_conf = np.max(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)
            class_pred = np.argmax(image_pred[:, 5: 5 + num_classes], axis=1, keepdims=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre)
            detections = np.concatenate((image_pred[:, :5], class_conf, class_pred.astype(np.float32)), axis=1)
            detections = detections[conf_mask]
            if detections.shape[0] == 0:
                continue

            # NMS
            boxes = detections[:, :4]
            scores = detections[:, 4] * detections[:, 5]
            if class_agnostic:
                indices = self.nms_numpy(boxes, scores, nms_thre)
            else:
                classes = detections[:, 6]
                indices = self.batched_nms_numpy(boxes, scores, classes, nms_thre)

            detections = detections[indices]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = np.concatenate((output[i], detections), axis=0)
        
            output[i][:, 0:4] = output[i][:, 0:4] / scale #modified
        return output

    @staticmethod
    def nms_numpy(boxes, scores, iou_threshold):
        # boxes: [N, 4], scores: [N]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def batched_nms_numpy(boxes, scores, classes, iou_threshold):
        # Perform NMS separately per class
        keep = []
        unique_classes = np.unique(classes)
        for cls in unique_classes:
            inds = np.where(classes == cls)[0]
            cls_keep = ObjectDetector.nms_numpy(boxes[inds], scores[inds], iou_threshold)
            keep.extend(inds[cls_keep])
        return keep
    
    def vis(self, img, boxes, scores, cls_ids, conf=0.5):
        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(cls_ids[i])
            score = scores[i]
            if score < conf:
                continue
            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            color = (self._COLORS[cls_id] * 255).astype(np.uint8).tolist()
            text = '{}:{:.1f}%'.format(self.CLASSES[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(self._COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX

            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

            txt_bk_color = (self._COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

        return img

def test():
    od = ObjectDetector("/home/teba/Programs/inrof2025/python/lib/masters.onnx")
    frame = cv2.imread("/home/teba/Programs/inrof2025/python/lib/chirobo.png")
    outputs = od.predict(frame)
    bboxes = outputs[:, 0:4]
    cls = outputs[:, 6]
    scores = outputs[:, 4] * outputs[:, 5]
    image = od.vis(frame, bboxes, scores, cls, 0.1)
    cv2.imshow("test", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test()