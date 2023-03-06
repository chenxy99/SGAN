import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="Selecting running mode (default: train)")
    parser.add_argument("--data_path", type=str,
                        default="/data/wikihow",
                        help="Data Path")
    parser.add_argument('--data_name', default='visualhow',
                        help='visualhow')
    parser.add_argument('--max_goal_length', default=20, type=int,
                        help='Maximum of goal length.')
    parser.add_argument('--max_problem_solving_step', default=10, type=int,
                        help='Maximum of problem solving steps.')
    parser.add_argument('--attention_step_num', type=int, default=10,
                        help='The number of the multimodal attention step in a single problem solving step')
    parser.add_argument('--tiny', action='store_true',
                        help='Use the tiny training set for verify.')
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument('--workers', default=4, type=int,
                        help='Number of data loader workers.')

    parser.add_argument('--num_epochs', default=10, type=int,
                        help='Number of training epochs.')
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="Weight decay")

    parser.add_argument('--fp16', action='store_true',
                        help='Use float 16 when training.')
    parser.add_argument('--optim', default='adam', type=str,
                        help='the optimizer')
    parser.add_argument('--lr_scheduler', default='cos', type=str,
                        help='the lr_scheduler')
    parser.add_argument('--backbone_warmup_epochs', type=int, default=5,
                        help='The number of epochs for warmup')
    parser.add_argument('--embedding_warmup_epochs', type=int, default=2,
                        help='The number of epochs for warming up the embedding layers')
    parser.add_argument('--backbone_lr_factor', default=0.01, type=float,
                        help='The lr factor for fine-tuning the backbone, it will be multiplied to the lr of '
                             'the embedding layers')
    parser.add_argument('--input_scale_factor', type=float, default=1,
                        help='The factor for scaling the input image')
    parser.add_argument('--train_loader_size', default=0, type=int,
                        help='the size of train loader')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='The number of epochs for warming up')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to logger.info and record the log.')

    parser.add_argument('--logger_name', default='../runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='../runs/runX ',
                        help='Path to save the model.')
    parser.add_argument('--eval_name', default='../runs/runX/eval',
                        help='Path to save the model.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--reset_start_epoch', action='store_true',
                        help='Whether restart the start epoch when load weights')

    # image encoder backbone
    parser.add_argument('--precomp_enc_type', default="backbone",
                        help='basic|backbone')
    parser.add_argument('--backbone_path', type=str, default='',
                        help='path to the pre-trained backbone net')
    parser.add_argument('--backbone_source', type=str, default='detector',
                        help='the source of the backbone model, detector|imagenet')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')

    # graph model
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--graph_layer_num', default=3, type=int,
                        help='Layer of Graph Convolutional Network.')
    parser.add_argument('--graph_head_num', default=1, type=int,
                        help='The head of the Graph Convolutional Network.')

    # loss
    parser.add_argument('--ohem_ratio', default=3, type=int,
                        help='Ratio of Online Hard Sample Mining.')
    parser.add_argument('--loss_type', type=str, default='bce',
                        help='the type of the loss function, bce|focalloss|ohem|ranking')
    parser.add_argument('--balance_loss', action='store_true',
                        help='for bce loss, determine whether to use balance loss')
    parser.add_argument('--two_stage', action='store_true',
                        help='determine whether to use two stage loss')
    parser.add_argument('--progressive_optimization', action='store_true',
                        help='Whether progressive to optimize the prediction in each GNN modules')
    parser.add_argument('--decay_factor', default=1.0, type=float,
                        help='The decay factor that balance the loss between different GNN modules')
    parser.add_argument('--task_completion', action='store_true',
                        help='Whether consider task completion in the model')
    parser.add_argument('--stepwise_task_completion', action='store_true',
                        help='Whether consider step wise task completion in the model')
    parser.add_argument('--stepwise_task_completion_weight', default=1.0, type=float,
                        help='the weight that balance the stepwise task completion loss and main loss')
    parser.add_argument('--similarity', default="jsd", type=str,
                        help='type of the similarity score for calculate the task completion, jsd|sim')
    parser.add_argument('--task_completion_weight', default=1.0, type=float,
                        help='the weight that balance the task completion loss and main loss')
    parser.add_argument('--attention_supervision', action='store_true',
                        help='Whether consider attention supervision in the model')
    parser.add_argument('--attention_supervision_weight', default=1.0, type=float,
                        help='the weight that balance the attention supervision loss and main loss')
    parser.add_argument('--margin', default=0.5, type=float, help='Rank loss margin.')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--vse_mean_warmup_epochs', type=int, default=0,
                        help='The number of warmup epochs using mean vse loss')
    parser.add_argument('--two_stage_warmup_epochs', type=int, default=0,
                        help='The number of warmup epochs using two stage loss')
    parser.add_argument('--include_second_stage', action='store_true',
                        help='Use two stage loss in the rank loss.')
    parser.add_argument('--contrastive_loss', action='store_true',
                        help='Use contrastive loss.')
    parser.add_argument('--contrastive_loss_weight', default=1.0, type=float,
                        help='the weight that balance the contrastive loss and main loss')
    parser.add_argument('--extend_graph', action='store_true',
                        help='Whether to extend the graph.')
    parser.add_argument('--self_dependency', action='store_true',
                        help='Whether to only consider dependency of the ground-truth problem solving step for the current goal.')
    parser.add_argument('--no_norm', action='store_true',
                        help='Do not normalize the embeddings.')
    parser.add_argument('--gamma', default=5.0, type=float,
                        help='the factor for the cos similarity of the attention')
    parser.add_argument("--dependency_type", nargs="+", default=["parallel", "others", "sequential"],
                        help='the type of the dependency used in data')
    parser.add_argument('--weighted_task_completion', action='store_true',
                        help='use the weights related the step connection number.')
    parser.add_argument('--interleave_validation', action='store_true',
                        help='interleave the order of validation/test set.')
    return parser
