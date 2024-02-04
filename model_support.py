import torch
import torch.nn as nn
import torch.nn.functional as F

# потери по контенту
class ContentMSE(nn.Module):
    def __init__(self):
        super(ContentMSE, self).__init__()
        self.target= None
    # делаем detach ибо обучать будем не их а входное изображение
    def set_target(self, target):
        self.target = target.detach()
    # перевод на device
    def to_device(self, device):
        if self.target is not None:
            self.target.to(device)
    def forward(self, input):
        loss = F.mse_loss(input, self.target)
        return loss
# потери по стилю
# Далее объявляются классы матрицы Грама и функции потерь для матрицы Грама
class GramMatrix(nn.Module):
    def forward(self, x):
        # a - размер батча, b - количество каналов,
        # c и d - размеры пространственных карт (высота и ширина)
        a, b, c, d = x.size()
        # каждая feature map представлена в виде вектора
        features = x.view(a * b, c * d)
        # произведение матрицы features на транспонированную features
        G = torch.mm(features, features.t())
        # матрица Грама, нормализованная по размеру входных данных
        return G.div(a*b*c*d)

class StyleMSE(nn.Module):
    def __init__(self):
        super(StyleMSE, self).__init__()
        self.target= None
    def set_target(self, target):
        # устанавливает целевую матрицу Грама для сравнения с выходными признаками
        # принимает target в качестве аргумента и сохраняет его, преобразуя в матрицу Грама
        # .detach() используется для предотвращения обновления target в процессе обратного распространения
        self.target = (GramMatrix()(target)).detach()

    def to_device(self, device):
        if self.target is not None :
            self.target.to(device)

    def forward(self, x):
        # используется среднеквадратичная ошибка (F.mse_loss)
        # между матрицей Грама входного изображения x и целевой матрицей Грама
        loss = F.mse_loss(GramMatrix()(x), self.target)
        return loss

# нормализация входного изображения
# представляет собой слой нормализации входного изображения для моделей VGG
class Normalization_for_VGG(nn.Module):
    def __init__(self):
        super(Normalization_for_VGG, self).__init__()

        #  средние значения и стандартные отклонения для нормализации по каналам RGB,
        # которые были использованы при обучении модели VGG
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    def forward(self, img):
        #  применяет нормализацию к входному изображению img, чтобы центрировать и нормализовать изображение
        return (img - self.mean) / self.std
    def to_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

# класс самой модели для переноса стиля
class model_work(nn.Module):
    def __init__(self, pool='max'):
        super(model_work, self).__init__()
        #  центрирует и нормализует входное изображение
        self.norm = Normalization_for_VGG()
        # Инициализация StyleMSE для вычисления style_loss на различных слоях
        self.style_loss_1 = StyleMSE()
        self.style_loss_2 = StyleMSE()
        self.style_loss_3 = StyleMSE()
        self.style_loss_4 = StyleMSE()
        self.style_loss_5 = StyleMSE()

        self.content_loss = ContentMSE()

        # Инициализация сверточного слоя conv_1 с 3 входными каналами (RGB) и 64 выходными каналами
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_4 = nn.Conv2d(128, 128,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_5 = nn.Conv2d(128, 256,  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

    # функция перемещения модели на device
    def to_device(self, device):
        self.norm.to_device(device)
        self.style_loss_1.to_device(device)
        self.style_loss_2.to_device(device)
        self.style_loss_3.to_device(device)
        self.style_loss_4.to_device(device)
        self.style_loss_5.to_device(device)
        self.content_loss.to_device(device)
        self.to(device)

        # fit_style, fit_content указывают, нужно ли настраивать модель на стиль или контент
    def forward(self, x, fit_style = False, fit_content = False):
        #  словарь out для хранения style_loss и content_loss
        out = {'style_loss':[], 'content_loss':[]}
        x = self.norm(x)
        x = self.conv_1(x)

        # если fit_style = True, модель проходит через сверточные слои, сохраняя
        # промежуточные значения для вычисления style_loss на различных уровнях

        # если fit_content = True, модель проходит через сверточные слои, сохраняя
        #  промежуточные значения для вычисления content_loss

        # если оба флага отключены, модель проходит через сверточные слои, вычисляя как style_loss,
        # так и content_loss на различных уровнях
        if fit_style:
            self.style_loss_1.set_target(x)
            x = self.conv_2(F.relu(x))
            self.style_loss_2.set_target(x)
            x = self.conv_3(self.pool1(F.relu(x)))
            self.style_loss_3.set_target(x)
            x = self.conv_4(F.relu(x))
            self.style_loss_4.set_target(x)
            x = self.conv_5(self.pool2(F.relu(x)))
            self.style_loss_5.set_target(x)
        elif fit_content:
            x = self.conv_2(F.relu(x))
            x = self.conv_3(self.pool1(F.relu(x)))
            x = self.conv_4(F.relu(x))
            self.content_loss.set_target(x)
        else:
            out['style_loss'].append (self.style_loss_1(x))
            x = self.conv_2(F.relu(x))
            out['style_loss'].append (self.style_loss_2(x))
            x = self.conv_3(self.pool1(F.relu(x)))
            out['style_loss'].append (self.style_loss_3(x))
            x = self.conv_4(F.relu(x))
            out['style_loss'].append (self.style_loss_4(x))
            out['content_loss'].append(self.content_loss(x))
            x = self.conv_5(self.pool2(F.relu(x)))
            out['style_loss'].append ( self.style_loss_5(x))

            return out
