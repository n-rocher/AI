import torch
import config
from config import args_setting
from model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import cv2
import os
import time
from torch.utils.data import Dataset

class Video_RoadSequenceDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {'data': self.data}

if __name__ == '__main__':
    args = args_setting()
    torch.manual_seed(args.seed)
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])

    # load model and weights
    model = generate_model(args)
    class_weight = torch.Tensor(config.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    pretrained_dict = torch.load(config.pretrained_path)
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # Get video in the video folder
    videos = os.listdir(config.video_path)

    for video in videos:
        cap = cv2.VideoCapture(config.video_path + "/" + video)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            break

        image_list = []

        while(cap.isOpened()):

            ret, frame = cap.read()
            photo_id = -1
            if ret == True:

                photo_id += 1

                resized = cv2.resize(frame, (config.img_width, config.img_height), interpolation=cv2.INTER_AREA)

                image_list.append(torch.unsqueeze(op_tranforms(resized), dim=0))
                if len(image_list) > 5:
                    image_list.pop(0)

                    cv2.imshow('Frame : ' + video, cv2.resize(frame, (600, 300), interpolation=cv2.INTER_AREA))

                    start = time.time()
                    data = torch.cat(image_list, 0)
                    data = Video_RoadSequenceDataset(data)
                    data_loader = torch.utils.data.DataLoader(data, batch_size=args.test_batch_size, shuffle=False, num_workers=1)
                    end = time.time()
                    print("DATALOADER : ", (end - start), "s")


                    start = time.time()
                    model.eval()
                    end = time.time()
                    print("EVAL : ", (end - start), "s")

                    feature_dic = []
                    with torch.no_grad():

                        print(data_loader)

                        for sample_batched in data_loader:
                            data = sample_batched["data"].to(device)
                            output, feature = model(data)
                            feature_dic.append(feature)
                            pred = output.max(1, keepdim=True)[1]
                            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1, -1, 3).numpy() * 255
                            img = Image.fromarray(img.astype(np.uint8))

                            data = torch.squeeze(data).cpu().numpy()
                            if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
                                data = np.transpose(data[-1], [1, 2, 0]) * 255
                            else:
                                data = np.transpose(data, [1, 2, 0]) * 255
                            data = Image.fromarray(data.astype(np.uint8))
                            rows = img.size[0]
                            cols = img.size[1]
                            start = time.time()
                            for i in range(0, rows):
                                for j in range(0, cols):
                                    img2 = (img.getpixel((i, j)))
                                    if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                                        data.putpixel((i, j), (234, 53, 57, 255))
                            end = time.time()
                            print("DOUBLE FOR LOOP : ", (end - start), "s")
                            data = data.convert("RGB")
                            cv2.imshow('Lane : ' + video, cv2.resize(np.array(img), (600, 300), interpolation=cv2.INTER_AREA))
                            cv2.imshow('Result : ' + video, cv2.resize(np.array(data), (600, 300), interpolation=cv2.INTER_AREA))
                            # data.save(config.save_path + "%s_data.jpg" % ("Video-" + str(photo_id)))  # red line on the original image
                            # img.save(config.save_path + "%s_pred.jpg" % ("Video-" + str(photo_id)))  # prediction result

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
