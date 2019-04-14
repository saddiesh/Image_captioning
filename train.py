from core.solver import CaptioningSolver
from core.model import CaptionGenerator
from core.utils import load_coco_data
# import tensorflow as tf

def main():
    # load train dataset
    # data = load_coco_data(data_path='./our_data', split='train')
    # word_to_idx = data['word_to_idx']
    # # load val dataset to print out bleu scores every epoch
    # test_data = load_coco_data(data_path='./our_data', split='test')
    #our train:
    data =load_coco_data(data_path='.\image_data_to_be_labeled\Object_feature\our_data', split='train')
    our_test = load_coco_data(data_path='.\image_data_to_be_labeled\Object_feature\our_data', split='train')
    word_to_idx = data['word_to_idx']
    model = CaptionGenerator(word_to_idx, dim_feature=[216, 512], dim_embed=512,
                                       dim_hidden=1024, n_time_step=26, prev2out=True,
                                                 ctx2out=True, alpha_c=1.0, selector=True, dropout=False)

    solver = CaptioningSolver(model, data, our_test, n_epochs=5000, batch_size=64, update_rule='adam',
                                          learning_rate=1e-4, print_every=1000, save_every=100, image_path='./image/',
                                    pretrained_model=None, model_path='model/our_train0414/', test_model='model/our_train0414/model-2000',
                                     print_bleu=False, log_path='log/')

    # solver.train()
    solver.test(our_test)
    # print(print(tf.test.is_gpu_available()))
    # print(tf.version)

if __name__ == "__main__":
    main()