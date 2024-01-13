import json
import torch
import socket
from PIL import Image
from functions import Functions
from exception import ConnectClosedException

import transformers

class Server():
    def __init__(self, 
                 device, 
                 host, 
                 port, 
                 model_params_root:str='./model_params', 
                 models_path:dict={
                     'retrieve': ('clip-vit-large-patch14-336', transformers.CLIPModel, transformers.CLIPProcessor), 
                     #'InstructBlip': ('instructblip-flan-t5-xl', transformers.InstructBlipForConditionalGeneration, transformers.InstructBlipProcessor), 
                     'vqa': ('llava-1.5-13b-hf', transformers.LlavaForConditionalGeneration, transformers.LlavaProcessor), 
                 }) -> None:
        self.functions = Functions(device, model_params_root, models_path)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.socket.listen(100)

        self.operate = {
            'test': self.test, 
            'test_receive_image': self.test_receive_image, 
        }
    
    def __del__(self):
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'functions'):
            del self.functions

    def run(self) -> None:
        while True:
            client_socket, addr = self.socket.accept()
            print(f'Connected with {addr}')
            while True:
                # 处理请求
                try:
                    request = self.receive_msg(client_socket)
                    self.deal_request(client_socket, request)
                except ConnectClosedException:
                    break
            print(f'Close connect with {addr}')
            client_socket.close()
    
    def decode_dict_msg(self, msg:bytes) -> dict:
        return json.loads(msg.decode('utf-8'))
    
    def encode_dict_msg(self, msg:dict) -> bytes:
        return json.dumps(msg).encode('utf-8')

    def receive_msg(self, 
                    client_socket:socket.socket) -> bytes:
        def receive(size):
            receive_size = 0
            total_msg = bytes()
            while receive_size < size:
                msg = client_socket.recv(size)
                if len(msg) == 0:
                    raise ConnectClosedException()
                total_msg += msg
                receive_size += len(msg) if size > 1024 else size
            client_socket.sendall('ok'.encode('utf-8'))
            return total_msg
        msg_size = self.decode_dict_msg(receive(1024))['msg_size']
        message = receive(msg_size)
        return message
    
    def send_msg(self, 
                 msg:bytes, 
                 client_socket:socket.socket) -> None:
        msg_metadata = self.encode_dict_msg({'msg_size':len(msg)})
        client_socket.sendall(msg_metadata)
        client_socket.recv(2)
        client_socket.sendall(msg)
        client_socket.recv(2)

    def receive_image(self, 
                      client_socket:socket.socket) -> Image:
        img_info = self.decode_dict_msg(self.receive_msg(client_socket))
        bytes_image = self.receive_msg(client_socket)
        img = Image.frombytes(img_info['mode'], img_info['size'], bytes_image)
        return img

    def deal_request(self, 
                     client_socket:socket.socket, 
                     request:bytes) -> None:
        request = self.decode_dict_msg(request)
        operate = request['operate']
        getattr(self, operate)(client_socket, request)
    
    def test(self, 
             client_socket:socket.socket, 
             request:dict) -> None:
        msg = self.receive_msg(client_socket)
        print(msg.decode('utf-8'))
        self.send_msg('get it'.encode('utf-8'), client_socket)
    
    def test_receive_image(self, 
                           client_socket:socket.socket, 
                           request:dict) -> None:
        img = self.receive_image(client_socket)
        img.show()
    
    def retrieve(self, 
                 client_socket:socket.socket, 
                 request:dict) -> None:
        image_num = request['image_num']
        top_k = request['top_k']
        batch_size = request['batch_size']
        text = self.receive_msg(client_socket).decode('utf-8')
        images = [self.receive_image(client_socket) for _ in range(image_num)]
        image_batchs = [images[i*batch_size:(i+1)*batch_size] for i in range(image_num // batch_size)]
        if image_num % batch_size != 0:
            image_batchs.append(images[-(image_num%batch_size):])

        logits_per_image = torch.cat([self.functions.retrieve(text, batch) for batch in image_batchs]).squeeze(1)
        #sim_indices = logits_per_image.sort(descending=True).indices
        #self.send_msg(self.encode_dict_msg({'top_k':sim_indices[:top_k].tolist()}), client_socket)
        self.send_msg(self.encode_dict_msg({'logits_per_image':logits_per_image.tolist()}), client_socket)
    
    def vqa(self, 
            client_socket:socket.socket, 
            request:dict) -> None:
        text_num = request['text_num']
        image_num = request['image_num']
        images_per_text = request['images_per_text']
        batch_size = request['batch_size']
        texts = [self.receive_msg(client_socket).decode('utf-8') for _ in range(text_num)]
        images = [self.receive_image(client_socket) for _ in range(image_num)]
        text_batchs = [texts[i*batch_size:(i+1)*batch_size] for i in range(text_num // batch_size)]
        if text_num % batch_size != 0:
            text_batchs.append(texts[-(text_num%batch_size):])
        images_ = []
        for image_num in images_per_text:
            images_.append(images[:image_num])
            images = images[image_num:]
        image_batchs = [sum(images_[i*batch_size:(i+1)*batch_size], start=[]) for i in range(image_num // batch_size)]
        if image_num % batch_size != 0:
            image_batchs.append(sum(images_[-(image_num%batch_size):], start=[]))
        
        answers = []
        for text_batch, image_batch in zip(text_batchs, image_batchs):
            answers += self.functions.vqa(texts=text_batch, images=image_batch)
        self.send_msg(self.encode_dict_msg({'answers':answers}), client_socket)

if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    #host = '192.168.0.110'
    host = 'localhost'
    port = 12345
    device='cuda'
    server = Server(device, host, port)
    server.run()