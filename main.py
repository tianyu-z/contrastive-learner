from comet_ml import Experiment, ExistingExperiment
import torch
from contrastive_learner import ContrastiveLearner
import model
import argparse
import dataloader_c as dataloader
import torchvision.transforms as transforms
from utils import init_net, upload_images, get_lr
import torch.optim as optim
import numpy as np
from loss import supervisedLoss, psnr, ssim

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_root",
    type=str,
    required=True,
    help="path to dataset folder containing train-test-validation folders",
)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    required=True,
    help="path to folder for saving checkpoints",
)
parser.add_argument(
    "--checkpoint", type=str, help="path of checkpoint for pretrained model"
)
parser.add_argument(
    "--train_continue", action="store_true", help="resuming from checkpoint."
)
parser.add_argument(
    "-it",
    "--init_type",
    default="",
    type=str,
    help="the name of an initialization method: normal | xavier | kaiming | orthogonal",
)

parser.add_argument(
    "--epochs", type=int, default=200, help="number of epochs to train. Default: 200."
)
parser.add_argument(
    "-tbs",
    "--train_batch_size",
    type=int,
    default=384,
    help="batch size for training. Default: 6.",
)
parser.add_argument(
    "-nw", "--num_workers", default=4, type=int, help="number of CPU you get"
)
parser.add_argument(
    "-vbs",
    "--validation_batch_size",
    type=int,
    default=384,
    help="batch size for validation. Default: 10.",
)
parser.add_argument(
    "-ilr",
    "--init_learning_rate",
    type=float,
    default=0.0001,
    help="set initial learning rate. Default: 0.0001.",
)
parser.add_argument(
    "--milestones",
    type=list,
    default=[100, 150],
    help="Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]",
)
parser.add_argument(
    "--progress_iter",
    type=int,
    default=100,
    help="frequency of reporting progress and validation. N: after every N iterations. Default: 100.",
)
parser.add_argument(
    "--logimagefreq", type=int, default=1, help="frequency of logging image.",
)
parser.add_argument(
    "--checkpoint_epoch",
    type=int,
    default=5,
    help="checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.",
)
parser.add_argument(
    "-wp", "--workspace", default="tianyu-z", type=str, help="comet-ml workspace"
)
parser.add_argument(
    "-dh", "--data_h", default=128, type=int, help="H of the data shape"
)
parser.add_argument(
    "-dw", "--data_w", default=128, type=int, help="W of the data shape"
)
parser.add_argument(
    "-pn",
    "--projectname",
    default="super-slomo",
    type=str,
    help="comet-ml project name",
)
parser.add_argument(
    "--nocomet", action="store_true", help="not using comet_ml logging."
)
parser.add_argument(
    "--cometid", type=str, default="", help="the comet id to resume exps",
)
parser.add_argument(
    "-rs",
    "--randomseed",
    type=int,
    default=2021,
    help="batch size for validation. Default: 10.",
)
parser.add_argument(
    "-pts",
    "--pretrainstage",
    type=int,
    default=1,
    help="batch size for validation. Default: 10.",
)
args = parser.parse_args()


class Trainer:
    def __init__(self, args=args):
        super().__init__()
        self.args = args
        # random_seed setting
        random_seed = args.randomseed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(random_seed)
        else:
            torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.pretrain_stage = self.args.pretrainstage
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.slomofc = model.Slomofc(
            self.args.data_h, self.args.data_w, self.device, self.pretrain_stage
        )
        self.slomofc.to(self.device)
        if self.pretrain_stage:
            self.learner = ContrastiveLearner(
                self.slomofc,
                image_size=128,
                hidden_layer="avgpool",
                use_momentum=True,  # use momentum for key encoder
                momentum_value=0.999,
                project_hidden=False,  # no projection heads
                use_bilinear=True,  # in paper, logits is bilinear product of query / key
                use_nt_xent_loss=False,  # use regular contrastive loss
                augment_both=False,  # in curl, only the key is augmented
            )
        if self.args.init_type != "":
            init_net(self.slomofc, self.args.init_type)
            print(self.args.init_type + " initializing slomo done!")
        if self.args.train_continue:
            if not self.args.nocomet and self.args.cometid != "":
                self.comet_exp = ExistingExperiment(
                    previous_experiment=self.args.cometid
                )
            elif not self.args.nocomet and self.args.cometid == "":
                self.comet_exp = Experiment(
                    workspace=self.args.workspace, project_name=self.args.projectname
                )
            else:
                self.comet_exp = None
            self.ckpt_dict = torch.load(self.args.checkpoint)
            self.slomofc.load_state_dict(self.ckpt_dict["model_state_dict"])
            self.args.init_learning_rate = self.ckpt_dict["learningRate"]
            if not self.pretrain_stage:
                self.optimizer = optim.Adam(
                    self.slomofc.parameters(), lr=self.args.init_learning_rate
                )
            else:
                self.optimizer = optim.Adam(self.learner.parameters(), lr=3e-4)
            self.optimizer.load_state_dict(self.ckpt_dict["opt_state_dict"])
            print("Pretrained model loaded!")
        else:
            # start logging info in comet-ml
            if not self.args.nocomet:
                self.comet_exp = Experiment(
                    workspace=self.args.workspace, project_name=self.args.projectname
                )
                # self.comet_exp.log_parameters(flatten_opts(self.args))
            else:
                self.comet_exp = None
            if not self.pretrain_stage:
                self.ckpt_dict = {
                    "trainLoss": {},
                    "valLoss": {},
                    "valPSNR": {},
                    "valSSIM": {},
                    "learningRate": {},
                    "epoch": -1,
                    "detail": "End to end Super SloMo.",
                    "trainBatchSz": self.args.train_batch_size,
                    "validationBatchSz": self.args.validation_batch_size,
                }
            else:
                self.ckpt_dict = {
                    "conLoss": {},
                    "learningRate": {},
                    "epoch": -1,
                    "detail": "Pretrain_stage of Super SloMo.",
                    "trainBatchSz": self.args.train_batch_size,
                }
            if not self.pretrain_stage:
                self.optimizer = optim.Adam(
                    self.slomofc.parameters(), lr=self.args.init_learning_rate
                )
            else:
                self.optimizer = optim.Adam(self.learner.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.args.milestones, gamma=0.1
        )
        # Channel wise mean calculated on adobe240-fps training dataset
        if not self.pretrain_stage:
            mean = [0.5, 0.5, 0.5]
            std = [1, 1, 1]
            self.normalize = transforms.Normalize(mean=mean, std=std)
            self.transform = transforms.Compose([transforms.ToTensor(), self.normalize])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        trainset = dataloader.SuperSloMo(
            root=self.args.dataset_root + "/train", transform=self.transform, train=True
        )
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
        )
        if not self.pretrain_stage:
            validationset = dataloader.SuperSloMo(
                root=self.args.dataset_root + "/validation",
                transform=self.transform,
                # randomCropSize=(128, 128),
                train=False,
            )
            self.validationloader = torch.utils.data.DataLoader(
                validationset,
                batch_size=self.args.validation_batch_size,
                num_workers=self.args.num_workers,
                shuffle=False,
            )
        ### loss
        if not self.pretrain_stage:
            self.supervisedloss = supervisedLoss()
            self.best = {
                "valLoss": 99999999,
                "valPSNR": -1,
                "valSSIM": -1,
            }
        else:
            self.best = {
                "conLoss": 99999999,
            }
        self.checkpoint_counter = int(
            (self.ckpt_dict["epoch"] + 1) / self.args.checkpoint_epoch
        )

    def train(self):
        for epoch in range(self.ckpt_dict["epoch"] + 1, self.args.epochs):
            print("Epoch: ", epoch)
            if not self.pretrain_stage:
                print("Training downstream task")
                print("Training epoch {}".format(epoch))
                _, _, train_loss = self.run_epoch(
                    epoch, self.trainloader, logimage=False, isTrain=True,
                )
                with torch.no_grad():
                    print("Validating epoch {}".format(epoch))
                    val_psnr, val_ssim, val_loss = self.run_epoch(
                        epoch, self.validationloader, logimage=True, isTrain=False,
                    )
                self.ckpt_dict["trainLoss"][str(epoch)] = train_loss
                self.ckpt_dict["valLoss"][str(epoch)] = val_loss
                self.ckpt_dict["valPSNR"][str(epoch)] = val_psnr
                self.ckpt_dict["valSSIM"][str(epoch)] = val_ssim
            else:
                print("Training pretrain task")
                conLoss = 0
                for trainIndex, data in enumerate(self.trainloader, 0):
                    loss = self.learner(data)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.learner.update_moving_average()  # update moving average of key encoder
                    conLoss += loss.item()
                conLoss = conLoss / len(self.trainloader)
                print(" Epoch: %4d  conLoss: %0.4f  " % (epoch, conLoss,))
                self.ckpt_dict["conLoss"][str(epoch)] = conLoss
                self.comet_exp.log_metric("conLoss", conLoss, epoch=epoch)

            self.ckpt_dict["learningRate"][str(epoch)] = get_lr(self.optimizer)
            self.ckpt_dict["epoch"] = epoch
            self.best = self.save_best(self.ckpt_dict, self.best, epoch)
            if (epoch % self.args.checkpoint_epoch) == self.args.checkpoint_epoch - 1:
                self.save()

    def save_best(self, current, best, epoch, pretrain_stage=True):
        save_best_done = False
        metrics = ["conLoss"] if pretrain_stage else ["valLoss", "valSSIM", "valPSNR"]
        for metric_name in metrics:
            if not save_best_done:
                if "Loss" in metric_name:
                    if best[metric_name] > current[metric_name][str(epoch)]:
                        best[metric_name] = current[metric_name][str(epoch)]
                        self.save(metric_name)
                        print(
                            "New Best "
                            + metric_name
                            + ": "
                            + str(best[metric_name])
                            + "saved"
                        )
                        save_best_done = True
                else:
                    if best[metric_name] < current[metric_name][str(epoch)]:
                        best[metric_name] = current[metric_name][str(epoch)]
                        self.save(metric_name)
                        print(
                            "New Best "
                            + metric_name
                            + ": "
                            + str(best[metric_name])
                            + "saved"
                        )
                        save_best_done = True
        return best

    @torch.no_grad()
    def save(self, save_metric_name=""):
        self.ckpt_dict["model_state_dict"] = self.slomofc.state_dict()
        self.ckpt_dict["opt_state_dict"] = self.optimizer.state_dict()
        file_name = (
            str(self.checkpoint_counter) if save_metric_name == "" else save_metric_name
        )
        model_name = (
            "/SuperSloMo" if not self.pretrain_stage else "/Pretrain_stage_SuperSloMo"
        )
        torch.save(
            self.ckpt_dict, self.args.checkpoint_dir + model_name + file_name + ".ckpt",
        )
        if save_metric_name == "":
            self.checkpoint_counter += 1

    ### Train and Valid
    def run_epoch(self, epoch, dataloader, logimage=False, isTrain=True):
        # For details see training.
        psnr_value = 0
        ssim_value = 0
        loss_value = 0
        if not isTrain:
            valid_images = []
        for index, all_data in enumerate(dataloader, 0):
            self.optimizer.zero_grad()
            (
                Ft_p,
                I0,
                IFrame,
                I1,
                g_I0_F_t_0,
                g_I1_F_t_1,
                FlowBackWarp_I0_F_1_0,
                FlowBackWarp_I1_F_0_1,
                F_1_0,
                F_0_1,
            ) = self.slomofc(all_data, pred_only=False, isTrain=isTrain)
            if (not isTrain) and logimage:
                if index % self.args.logimagefreq == 0:
                    valid_images.append(
                        255.0
                        * I0.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * IFrame.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * I1.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
                    valid_images.append(
                        255.0
                        * Ft_p.cpu()[0]
                        .resize_(1, 1, self.args.data_h, self.args.data_w)
                        .repeat(1, 3, 1, 1)
                    )
            # loss
            loss = self.supervisedloss(
                Ft_p,
                IFrame,
                I0,
                I1,
                g_I0_F_t_0,
                g_I1_F_t_1,
                FlowBackWarp_I0_F_1_0,
                FlowBackWarp_I1_F_0_1,
                F_1_0,
                F_0_1,
            )
            if isTrain:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            loss_value += loss.item()

            # metrics
            psnr_value += psnr(Ft_p, IFrame, outputTensor=False)
            ssim_value += ssim(Ft_p, IFrame, outputTensor=False)

        name_loss = "TrainLoss" if isTrain else "ValLoss"
        itr = int(index + epoch * (len(dataloader)))
        if self.comet_exp is not None:
            self.comet_exp.log_metric(
                "PSNR", psnr_value / len(dataloader), step=itr, epoch=epoch
            )
            self.comet_exp.log_metric(
                "SSIM", ssim_value / len(dataloader), step=itr, epoch=epoch
            )
            self.comet_exp.log_metric(
                name_loss, loss_value / len(dataloader), step=itr, epoch=epoch
            )
            if logimage:
                upload_images(
                    valid_images,
                    epoch,
                    exp=self.comet_exp,
                    im_per_row=4,
                    rows_per_log=int(len(valid_images) / 4),
                )
        print(
            " Loss: %0.6f  Iterations: %4d/%4d  ValPSNR: %0.4f  ValSSIM: %0.4f "
            % (
                loss_value / len(dataloader),
                index,
                len(dataloader),
                psnr_value / len(dataloader),
                ssim_value / len(dataloader),
            )
        )
        return (
            (psnr_value / len(dataloader)),
            (ssim_value / len(dataloader)),
            (loss_value / len(dataloader)),
        )


if __name__ == "__main__":
    trainer = Trainer(args)
    trainer.train()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([transforms.ToTensor()])
# trainset = dataloader.SuperSloMo(
#     root=args.dataset_root + "/train", transform=transform, train=True
# )
# trainloader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=args.train_batch_size,
#     num_workers=args.num_workers,
#     shuffle=True,
# )

# validationset = dataloader.SuperSloMo(
#     root=args.dataset_root + "/validation",
#     transform=transform,
#     randomCropSize=(128, 128),
#     train=False,
# )
# validationloader = torch.utils.data.DataLoader(
#     validationset,
#     batch_size=args.validation_batch_size,
#     num_workers=args.num_workers,
#     shuffle=False,
# )


# # resnet = models.resnet50(pretrained=True)
# learner = ContrastiveLearner(
#     slomofc,
#     image_size=128,
#     hidden_layer="avgpool",
#     use_momentum=True,  # use momentum for key encoder
#     momentum_value=0.999,
#     project_hidden=False,  # no projection heads
#     use_bilinear=True,  # in paper, logits is bilinear product of query / key
#     use_nt_xent_loss=False,  # use regular contrastive loss
#     augment_both=False,  # in curl, only the key is augmented
# )

# opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


# def sample_batch_images():
#     return torch.randn(20, 3, 256, 256)


# for _ in range(100):
#     for trainIndex, data in enumerate(trainloader, 0):
#         loss = learner(data)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         learner.update_moving_average()  # update moving average of key encoder
#         print("success")
