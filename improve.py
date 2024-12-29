import os
import torch
from torchvision import transforms
from PIL import Image
import sys
import logging

logging.basicConfig(level=logging.INFO)

class MDSR(torch.nn.Module):
    def __init__(self):
        super(MDSR, self).__init__()
        
        self.sub_mean = torch.nn.Conv2d(3, 3, kernel_size=1)
        self.add_mean = torch.nn.Conv2d(3, 3, kernel_size=1)
        
        self.pre_process = torch.nn.ModuleList()
        for i in range(3):
            block = torch.nn.ModuleList()
            for j in range(2):
                sub_block = torch.nn.ModuleDict({
                    'body': torch.nn.Sequential(
                        torch.nn.Conv2d(64, 64, kernel_size=5, padding=2),
                        torch.nn.ReLU(True),
                        torch.nn.Conv2d(64, 64, kernel_size=5, padding=2)
                    )
                })
                block.append(sub_block)
            self.pre_process.append(block)
        
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        )
        
        self.body = torch.nn.ModuleList()
        for i in range(16):
            block = torch.nn.ModuleDict({
                'body': torch.nn.Sequential(
                    torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                )
            })
            self.body.append(block)
        self.body.append(torch.nn.Conv2d(64, 64, kernel_size=3, padding=1))
        
        from collections import OrderedDict
        
        self.upsample = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 256, kernel_size=3, padding=1)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 576, kernel_size=3, padding=1)
            ),
            torch.nn.Sequential(
                OrderedDict([
                    ('0', torch.nn.Conv2d(64, 256, kernel_size=3, padding=1)),
                    ('1', torch.nn.PixelShuffle(2)),
                    ('2', torch.nn.Conv2d(64, 256, kernel_size=3, padding=1))
                ])
            )
        ])
        
        self.tail = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = x * 255.0
        
        x = self.sub_mean(x)
        x = self.head(x)
        
        # Pre-process с очисткой градиентов
        for block in self.pre_process:
            for sub_block in block:
                res = x
                x = sub_block['body'](x)
                x = x + res
                torch.cuda.empty_cache()
        
        residual = x
        for i, block in enumerate(self.body[:-1]):
            res = x
            x = block['body'](x)
            x = x + res
            if i % 4 == 0:
                torch.cuda.empty_cache()
        
        x = self.body[-1](x)
        x = x + residual
        
        main_flow = x
        x = self.upsample[-1](main_flow)
        x = torch.nn.functional.pixel_shuffle(x, 2)
        
        x = self.tail(x)
        x = self.add_mean(x)
        
        x = x / 255.0
        return torch.clamp(x, 0, 1)

def load_image(image_path, max_size=1440):
    image = Image.open(image_path).convert('RGB')
    
    width, height = image.size
    aspect_ratio = width / height
    
    if width > max_size:
        width = max_size
        height = int(width / aspect_ratio)
    if height > max_size:
        height = max_size
        width = int(height * aspect_ratio)
    
    if width != image.size[0] or height != image.size[1]:
        image = image.resize((width, height), Image.LANCZOS)
    
    tensor = transforms.ToTensor()(image)
    return tensor.unsqueeze(0)

def save_image(tensor, output_path):
    tensor = tensor.squeeze(0)
    
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = tensor.clamp(0, 1)
    
    image = transforms.ToPILImage()(tensor)
    image.save(output_path, quality=95)
    logging.info(f'Изображение успешно сохранено: {output_path}')

def main(input_image_path, output_image_path):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Используется устройство: {device}")
        
        model_path = os.path.expanduser('~/models/mdsr/mdsr_baseline-a00cab12.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        model = MDSR().to(device)
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            logging.info("Модель успешно загружена")
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке весов модели: {e}")
        
        model.eval()
        
        logging.info("Загрузка входного изображения...")
        input_image = load_image(input_image_path, max_size=4096).to(device)
        
        logging.info("Обработка изображения...")
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.backends.cudnn.benchmark = True
            
            output_image = model(input_image)
            output_image = output_image.cpu()
        
        logging.info("Сохранение результата...")
        save_image(output_image, output_image_path)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        logging.error("Использование: python improve.py input_image output_image")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    
    main(input_image_path, output_image_path)

