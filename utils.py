import os
import os.path
import torch


class saveData():
    def __init__(self, args):
        self.args = args
        self.result_SR_Dir = args.result_SR_Dir # os.path.join(args.saveDir, args.net_name)
        if not os.path.exists(self.result_SR_Dir):
            os.makedirs(self.result_SR_Dir)
        self.save_dir_model = args.model_savepath
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)
        if os.path.exists(self.save_dir_model + '/log.txt'):
            self.logFile = open(self.save_dir_model + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir_model + '/log.txt', 'w')

    def save_model(self, model, epoch, mode_path):
        # torch.save(model.state_dict(), './VDSR/model_vdsr/vdsr{}'.format(epoch)) model_rdn/model_vdsr
        torch.save(model.state_dict(), mode_path + '\RDN_{}'.format(epoch))
        # torch.save(
        #    model.state_dict(),'./result/face_x8_small/model16_{}'.format(epoch)) #vdsr
        #    # self.save_dir_model + '/model_{}_{}.pt'.format(4_10,epoch))

    def save_log(self, log):
        self.logFile.write(log + '\n')

    def load_model(self, model, model_path):
        model.load_state_dict(torch.load(model_path))
        print("model from :{}".format(model_path))
        return model
