import numpy as np
import os
from configparser import ConfigParser
from generator_test import AugmentedImageSequence2
from models.keras import ModelFactory
from sklearn.metrics import roc_auc_score
from utilities import get_sample_counts

def main():

    # default config
    output_dir = './outputs'
    base_model_name = 'InceptionResNetV2'
    class_names = Atelectasis,Cardiomegaly,Effusion,Infiltration,Mass,Nodule,Pneumonia,Pneumothorax,Consolidation,Edema,Emphysema,Fibrosis,Pleural_Thickening,Hernia
    image_source_dir = './Images'
    image_dimension = 341
    batch_size = 16
    test_steps = 1
    use_best_weights = True
    output_weights_name = weights.h5
    weights_path = '
    best_weights_path = './outputs/best_auroc.h5

    # get test sample count
    test_counts, _ = get_sample_counts(output_dir, "testt", class_names)

    # compute steps
    if test_steps == "auto":
        test_steps = int(test_counts / batch_size)
    else:
        try:
            test_steps = int(test_steps)
        except ValueError:
            raise ValueError(f"""
                test_steps: {test_steps} is invalid,
                please use 'auto' or integer.
                """)
    print(f"** test_steps: {test_steps} **")

    print("** load model **")
    if use_best_weights:
        print("** use best weights **")
        model_weights_path = best_weights_path
    else:
        print("** use last weights **")
        model_weights_path = weights_path
    model_factory = ModelFactory()
    model = model_factory.get_model(
        class_names,
        model_name=base_model_name,
        use_base_weights=False,
        weights_path=model_weights_path)

    print("** load test generator **")
    test_sequence = AugmentedImageSequence2(
        dataset_csv_file=os.path.join(output_dir, "testt.csv"),
        class_names=class_names,
        source_image_dir=image_source_dir,
        batch_size=batch_size,
        target_size=(image_dimension, image_dimension),
        augmenter=None,
        steps=test_steps,
        shuffle_on_epoch_end=False,
    )
    print("** make prediction **")
    y_hat = model.predict_generator(test_sequence, verbose=1)
    y = test_sequence.get_y_true()
    np.save('y_hat_val.npy',y_hat)
    np.save('y_val.npy',y)
    test_log_path = "./outputs/val.log")
    print(f"** write log to {test_log_path} **")
    aurocs = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(f"{class_names[i]}: {score}\n")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write(f"mean auroc: {mean_auroc}\n")
        print(f"mean auroc: {mean_auroc}")


if __name__ == "__main__":
    main()
