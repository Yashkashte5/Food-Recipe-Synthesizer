import ast
import logging
import os
import sys
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
)

from data_utils import (
    filter_by_lang_regex,
    filter_by_steps,
    filter_by_length,
    filter_by_item,
    filter_by_num_sents,
    filter_by_num_tokens,
    normalizer
)

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    """
    Arguments to which dataset we are going to set up.
    """

    output_dir: str = field(
        default=".",
        metadata={"help": "The output directory where the config will be written."},
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_data_dir: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


def main():
    parser = HfArgumentParser([DataArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        data_args = parser.parse_args_into_dataclasses()[0]

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.info(f"Preparing the dataset")

    if data_args.dataset_name is not None:
        dataset = load_dataset(
            data_args.dataset_name,
            data_dir=data_args.dataset_data_dir,
            cache_dir=data_args.cache_dir
        )
    else:
        dataset = load_dataset(
            data_args.dataset_name,
            cache_dir=data_args.cache_dir
        )

    def cleaning(text, item_type="ner"):
        # NOTE: DO THE CLEANING LATER
        text = normalizer(text, do_lowercase=True)
        return text

    def recipe_preparation(item_dict):
        ner = item_dict["ner"]
        title = item_dict["title"]
        ingredients = item_dict["ingredients"]
        steps = item_dict["directions"]

        conditions = []
        conditions += [filter_by_item(ner, 2)]
        conditions += [filter_by_length(title, 4)]
        conditions += [filter_by_item(ingredients, 2)]
        conditions += [filter_by_item(steps, 2)]
        # conditions += filter_by_steps(" ".join(steps))

        if not all(conditions):
            return None

        ner = ", ".join(ner)
        ingredients = " <sep> ".join(ingredients)
        steps = " <sep> ".join(steps)

        # Cleaning
        ner = cleaning(ner, "ner")
        title = cleaning(title, "title")
        ingredients = cleaning(ingredients, "ingredients")
        steps = cleaning(steps, "steps")

        return {
            "inputs": ner,
            # "targets": f"title: {title} <section> ingredients: {ingredients} <section> directions: {steps}"
            "targets": f"title: {title} <section> ingredients: {ingredients} <section> directions: {steps}"
        }

    if len(dataset.keys()) > 1:
        for subset in dataset.keys():
            data_dict = []
            for item in tqdm(dataset[subset], position=0, total=len(dataset[subset])):
                item = recipe_preparation(item)
                if item:
                    data_dict.append(item)

            data_df = pd.DataFrame(data_dict)
            logger.info(f"Preparation of [{subset}] set consists of {len(data_df)} records!")

            output_path = os.path.join(data_args.output_dir, f"{subset}.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data_df.to_csv(output_path, sep="\t", encoding="utf-8", index=False)
            logger.info(f"Data saved here {output_path}")
    else:
        data_dict = []
        subset = list(dataset.keys())[0]
        for item in tqdm(dataset[subset], position=0, total=len(dataset[subset])):
            item = recipe_preparation(item)
            if item:
                data_dict.append(item)

        data_df = pd.DataFrame(data_dict)

        logger.info(f"Preparation - [before] consists of {len(dataset[subset])} records!")
        logger.info(f"Preparation - [after]  consists of {len(data_df)} records!")

        train, test = train_test_split(data_df, test_size=0.05, random_state=101)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        logger.info(f"Preparation of [train] set consists of {len(train)} records!")
        logger.info(f"Preparation of [test] set consists of {len(test)} records!")

        os.makedirs(data_args.output_dir, exist_ok=True)
        train.to_csv(os.path.join(data_args.output_dir, "train.csv"), sep="\t", encoding="utf-8", index=False)
        test.to_csv(os.path.join(data_args.output_dir, "test.csv"), sep="\t", encoding="utf-8", index=False)
        logger.info(f"Data saved here {data_args.output_dir}")


if __name__ == '__main__':
    main()
