import time
import pickle
import torch
import torch.optim as optim
from pathlib import Path
from PIL import Image
import copy
import torchvision.transforms as transforms
import logging

from model_support import model_work


class Style_Transfer():
    def __init__(self):
        self.DEVICE = None
        self.model = None
        self.size = None
        logging.basicConfig(filename='style_transfer.log', level=logging.INFO)

    def getting_images(self, path, chat_id):
        print('getting_images  self.size ',self.size)
        transformer = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size),
            transforms.ToTensor()])
        # читаем картинки
        style_image = Image.open(path/str(chat_id)/"style.jpg")
        content_image = Image.open(path/str(chat_id)/"content.jpg")
        # добавляем ось для соответствия входному размеру сети
        style_image = transformer(style_image).unsqueeze(0)\
                            .to(torch.float)
        content_image = transformer(content_image).unsqueeze(0)\
                            .to(torch.float)
        return style_image, content_image

    def style_transfer (self, style_image, content_image, num_steps= 25,
                       style_weight=70000, content_weight=1):
        self.model.to_device(self.DEVICE)
        self.model.eval()
        self.model(style_image.to(self.DEVICE), fit_style = True)
        self.model(content_image.to(self.DEVICE), fit_content = True)
        self.model.requires_grad_(False) # параметры не меняются

        input_image = content_image.to(self.DEVICE)
        input_image.requires_grad_(True)

        best_image = input_image.to('cpu')
        best_score = float('inf')

        # словарь для лоссов
        losses = {'style': [], 'content' : []}

        optimizer = optim.LBFGS([input_image])
        print('Optimizing..')

        def closure():
            with torch.no_grad():
                input_image.clamp_(0, 1)
            optimizer.zero_grad()
            loss_dict = self.model(input_image)
            style_score = sum(loss_dict['style_loss'])
            content_score = sum(loss_dict['content_loss'])
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()

            losses['style'].append(style_score.item())
            losses['content'].append(content_score.item())
            return style_score + content_score

        for i in range (num_steps):
            optimizer.step(closure)
            # лучший результат
            if losses['style'][-1] < best_score:
                best_image =  input_image.to('cpu')
                best_score = losses['style'][-1]
            if i % 3 == 0:
                logging.info(f"Step {i}, Style Loss: {losses['style'][-1]}, Content Loss: {losses['content'][-1]}")

        with torch.no_grad():
            best_image.clamp_(0, 1)
        self.model.to_device('cpu')
        optimizer = None
        loss = None
        loss_dict =None
        style_score = None
        content_score = None
        input_image = None
        torch.cuda.empty_cache()
        return best_image, losses

    def transfering(self, path , chat_id ,
                    size = 512, num_steps=25,
                    style_weight=50000, content_weight=1):

        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.size = size
        else:
            self.size = 128
        style_image, content_image = self.getting_images(path, chat_id)
        if self.model is None:
            self.model = model_work()
            # загружает веса модели self.model из файла 'vgg_weights.pth'
            self.model.load_state_dict(torch.load(Path(__file__).parent.resolve()/'vgg_weights.pth'))
        # перенос стиля
        best_image, losses = self.style_transfer(style_image, content_image, num_steps = num_steps,
                       style_weight=style_weight, content_weight=content_weight)
        # пропорции картинки для восстановления
        with open (path/str(chat_id)/'content_proportion.txt', 'r') as file:
            proportion = float(file.readlines()[0])
        # ширина и высота выходной картинки
        width, height = size, size
        if proportion >1:
            width = int(size*proportion)

        else:
            height = int(size / proportion)


        topiller = transforms.ToPILImage()
        output_img = topiller(best_image.squeeze(0))\
                    .resize((width, height))
        # сохраняем результат
        time_marker = str(int(time.time()))
        output_img.save(path /str(chat_id)/
                        ('out_'
                        + time_marker
                        + '.jpg'))
        # сохраним историю лоссов
        path_to_losses =(path /str(chat_id)/
                         ('losses_'
                        + time_marker
                        + '.pkl'))
        with open(path_to_losses, 'wb') as file:
            pickle.dump(losses, file)
        # очистка GPU
        torch.cuda.empty_cache()
        return output_img
