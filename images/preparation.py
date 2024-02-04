import requests
import os
from PIL import Image
from io import BytesIO

# Загружает изображение по file_path
# Изменяет размер изображения до заданного size x size
# Сохраняет изображение и пропорцию в каталоге для конкретного чата под именем category.jpg и category_proportion.txt
def change_and_save_image(file_path, size , chat_id , category, work_dir):
    try :
        response  = requests.get(file_path)
        img = Image.open(BytesIO(response.content))
        proportion = img.size[0]/img.size[1]
        print ('proportion from change_and_save_image', proportion)
        img=img.resize((size, size))
        save_images = work_dir/str(chat_id)
        if not os.path.exists(save_images ):
                os.makedirs(save_images)

        img.save(save_images/(category +'.jpg'))
        with open (save_images/(category + '_proportion.txt'), 'w') as file:
            file.write(str(proportion))
        return 0
    # текст ошибки в случае неудачи
    except Exception as e :
        return str(e)

# Открывает ранее сохраненное изображение из каталога чата.
# Восстанавливает изображение с учетом сохраненной пропорции.
# Преобразует изображение в формат, который бот может отправить в Telegram.
def open_and_repair_image( size, chat_id , category, work_dir ):
    save_images = work_dir /str(chat_id)
    img = Image.open(save_images /( category + '.jpg'))
    with open(save_images/(category + '_proportion.txt'), 'r') as file:
        proportion = float(file.readlines()[0])
        print ('proportion from open_and_repair_image', proportion)
    img = img.resize( (int(size*proportion),size))
    # Преобразует изображение в формат, который бот может отправить в Telegram.
    bio = BytesIO()
    bio.name = 'image.jpeg'
    img.save(bio, 'JPEG')
    bio.seek(0)
    return bio
