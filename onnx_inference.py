import onnxruntime
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from option import get_args

opt = get_args()


class ONNXInference:
    def __init__(self, onnx_path, loadsize=224):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.loadsize = loadsize

        self.transform = transforms.Compose([
            transforms.Resize((loadsize, loadsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.6786, 0.6413, 0.6605], [0.2599, 0.2595, 0.2569])
        ])

    def predict_single(self, image_path):
        image = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)

        img_numpy = img_tensor.numpy()
        output = self.session.run([self.output_name], {self.input_name: img_numpy})
        output = np.array(output[0])
        predicted_class = np.argmax(output, axis=1)[0]
        confidence = np.max(output, axis=1)[0]

        return predicted_class, confidence

    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            pred_class, conf = self.predict_single(path)
            results.append((path, pred_class, conf))
        return results


def main():
    args = opt
    class_names = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

    onnx_model = ONNXInference(args.onnx_path, args.loadsize)

    pred_class, confidence = onnx_model.predict_single(args.test_img_path)
    print('Predicted: {}  Confidence: {:.4f}'.format(class_names[pred_class], confidence))


if __name__ == '__main__':
    main()