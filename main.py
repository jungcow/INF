import sys
import importlib
import options
from model.modelwrapper import ModelWrapper

if __name__=="__main__":
    opt_list = options.set(arg=sys.argv[1:])

    model_list = []
    for i in range(len(opt_list)):
        model = importlib.import_module("model.{}".format(opt_list[i].model))
        model_list.append(model)

    print(f"{len(model_list)} models found!")
    if len(opt_list) == 1:
        opt = opt_list[0]
        m = model.Model(opt)
        m.build_network(opt)
        m.set_optimizer(opt)
        m.create_dataset(opt)
        m.train(opt)

        m.end_process(opt_list)
    else:
        m = ModelWrapper(model_list=model_list, opt_list=opt_list)
        m.build_network(opt_list=opt_list)
        m.set_optimizer(opt_list=opt_list)
        m.create_dataset(opt_list=opt_list)
        m.train(opt_list=opt_list)
        m.end_process(opt_list)
