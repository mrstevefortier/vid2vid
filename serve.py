### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import ServeOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

import falcon
import png
from wsgiref import simple_server

opt = ServeOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
if opt.dataset_mode == 'temporal':
    opt.dataset_mode = 'test'

class Inference(object):
    def __init__(self):
        self.data_loader = CreateDataLoader(opt)
        self.dataset = data_loader.load_data()
        self.model = create_model(opt)
        self.visualizer = Visualizer(opt)
        self.input_nc = 1 if opt.label_nc != 0 else opt.input_nc
        self.save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

    def on_get(self, req, resp):
        imagePath = "/home/stevefortier/Desktop/WEvrDg.png" #model.infer(json, size)
        modelName = req.get_param("modelName", required=True)
        entityId = req.get_param("entityName", required=True)
        sizeString = req.get_param("size", required=True)
        sizeStringArray = sizeString.split(',')
        size = (int(sizeStringArray[0]), int(sizeStringArray[1]))
        
        # append new sample



        # Old thing
        print('Doing %d frames' % len(dataset))
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break    
            if data['change_seq']:
                self.model.fake_B_prev = None

            _, _, height, width = data['A'].size()
            A = Variable(data['A']).view(1, -1, self.input_nc, height, width)
            B = Variable(data['B']).view(1, -1, opt.output_nc, height, width) if len(data['B'].size()) > 2 else None
            inst = Variable(data['inst']).view(1, -1, 1, height, width) if len(data['inst'].size()) > 2 else None
            generated = self.model.inference(A, B, inst)
            
            if opt.label_nc != 0:
                real_A = util.tensor2label(generated[1], opt.label_nc)
            else:
                c = 3 if opt.input_nc == 3 else 1
                real_A = util.tensor2im(generated[1][:c], normalize=False)    
                
            visual_list = [('real_A', real_A), 
                           ('fake_B', util.tensor2im(generated[0].data[0]))]
            visuals = OrderedDict(visual_list) 
            img_path = data['A_path']
            print('process image... %s' % img_path)
            self.visualizer.save_images(save_dir, visuals, img_path)


        json = "" # req.something

        resp.content_type = falcon.MEDIA_PNG
        resp.status = falcon.HTTP_200
        with open(imagePath, "rb") as pngFile:
            resp.body = pngFile.read()
        
class Server:
    def __init__(self, port):
        self.port = port

    def serve(self):
        ## Real thing to call
        app = falcon.API()
        app.req_options.auto_parse_form_urlencoded=True

        inference = Inference()
        app.add_route('/infer', inference)

        print("Serving on port {}".format(self.port))
        httpd = simple_server.make_server('127.0.0.1', self.port, app)
        httpd.serve_forever()


server = Server(opt.port)
server.serve()
