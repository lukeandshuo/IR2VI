import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import ntpath
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.resize_or_crop = None # no image preprocessing in test code
opt.serial_batches = False  # no shuffle
opt.no_flip = True  # no flip
opt.phase = "test"
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    img_path = model.get_image_paths()
    save_dir = "/data/Sensiac/SensiacNight/Imagery/background_with_object_v3(resnet_9blocks)/images/"
    model.test()
    visuals = model.get_current_visuals()

    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images(webpage, visuals, img_path)

    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # visualizer.save_fakeB_images(save_dir, visuals, img_path)
webpage.save()
