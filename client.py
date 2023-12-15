import json
import socket
from PIL import Image
from exception import ConnectClosedException

class Client():
    def __init__(self, 
                 host:str, 
                 port:int=11451) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
    
    def __del__(self):
        self.socket.close()
    
    def decode_dict_msg(self, msg:bytes) -> dict:
        return json.loads(msg.decode('utf-8'))
    
    def encode_dict_msg(self, msg:dict) -> bytes:
        return json.dumps(msg).encode('utf-8')
    
    def send_msg(self, msg:bytes) -> None:
        msg_metadata = self.encode_dict_msg({'msg_size':len(msg)})
        self.socket.sendall(msg_metadata)
        self.socket.recv(2)
        self.socket.sendall(msg)
        self.socket.recv(2)
    
    def receive_msg(self) -> bytes:
        def receive(size):
            msg = self.socket.recv(size)
            if len(msg) == 0:
                raise ConnectClosedException()
            self.socket.sendall('ok'.encode('utf-8'))
            return msg
        msg_size = self.decode_dict_msg(receive(1024))['msg_size']
        message = receive(msg_size)
        return message

    def send_operate(self, 
                     operate:str, 
                     **kwargs) -> None:
        kwargs.update({'operate':operate})
        self.send_msg(self.encode_dict_msg(kwargs))

    def send_image(self, image:Image) -> None:
        self.send_msg(self.encode_dict_msg({
            'mode':image.mode, 
            'size':image.size
            }))
        self.send_msg(image.tobytes())
    
    def test(self) -> None:
        self.send_operate('test')
        self.send_msg('Hello'.encode('utf-8'))
        response = self.receive_msg().decode('utf-8')
        print(response)
    
    def test_image(self) -> None:
        self.send_operate('test_receive_image')
        img = Image.open('E:\Pictures\Pictures\Diyidan\g6H378aGhFle6fHE.jpg')
        self.send_image(img)
    
    def retrieve(self, 
                 text:str, 
                 images:list, 
                 top_k:int=1, 
                 batch_size=128) -> list:
        self.send_operate('retrieve', 
                          image_num=len(images), 
                          top_k=top_k, 
                          batch_size=batch_size)
        self.send_msg(text.encode('utf-8'))
        for i in range(len(images)):
            self.send_image(images[i])
        top_k_indices = self.decode_dict_msg(self.receive_msg())['top_k']
        return top_k_indices

    def vqa(self, 
            question:str, 
            image:Image) -> str:
        pass

if __name__ == '__main__':
    host = '192.168.0.110'
    port = 12345
    client = Client(host, port)
    
    import os
    import random
    seed = 42
    random.seed(seed)
    root = 'E:\Pictures\pixiv'
    img_num = 100
    text = 'Which one has two girls?'
    images_path = []
    for _, _, files in os.walk(root):
        images_path = random.sample(files, img_num)
        break
    images = [Image.open(os.path.join(root, path)) for path in images_path]
    top_k_indices = client.retrieve(text, images, top_k=10)
    for idx in top_k_indices:
        images[idx].show()