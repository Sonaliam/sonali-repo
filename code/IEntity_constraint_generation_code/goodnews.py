import argparse
import logging
import sys
import yaml
import json

from utils import allowed_image_file
from newsanalyzer import NewsAnalyzer
from metrics import cossim

def calculate_cms(self, image_embeddings, entity_embeddings):
    #cossim
    entity_sims = cossim(
        np.asarray(image_embeddings, dtype=np.float32), np.asarray(entity_embeddings, dtype=np.float32)
        )
    if self.config["operator"] == "max":
        return np.max(entity_sims)
    if self.config["operator"].startswith("q") and len(self.config["operator"]) > 1:  
        return np.quantile(entity_sims, float(self.config["operator"][1:]) / 100)
    logging.warning(f"Unknown operator {self.config['operator']}. Using max instead ...")
    return np.max(entity_sims)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script to quantify the cross-modal consistency of entities in image-text pairs."
    )

    # required arguments
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "-w",
        "--wikifier_key",
        type=str,
        required=True,
        help="Your Wikifier key from http://www.wikifier.org/register.html",
    )

    # optional arguments
    parser.add_argument("-v", "--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        required=False,
        default="en",
        choices=["en", "de"],
        help="Language of the input text",
    )

    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    # initialize newsanalyzer
    na = NewsAnalyzer(wikifier_key=args.wikifier_key, config=config)


    constraints ={}
    l=0
    im_cap = '../news_images/data/article_caption.json'         
    with open(im_cap) as train_file1:
        im_name_cap= json.load(train_file1)
    len(im_name_cap)
    keys_list = []
    for k,v in im_name_cap.items():
        temp ={}
        #m =0
        for key in v['images'].keys():
            image = '../datasets/goodnews/images_processed/'+k+'_'+key+'.jpg'
            text = v['article']

            # check image input
            if not allowed_image_file(image):
                logging.error("Image extension unknown. Exiting ...")
            return 0
            keys_list.append(k) 
            try:
                entity_embeddings, entities_cms = na.get_entity_cms(image_file=image, text=text, language=args.language)

                if len(entity_embeddings) != 0:      
                    if self.config["face_clustering"] and entity["type"] == "PERSON":
                        entity_embeddings = agglomerative_clustering(entity_embeddings)
                    logging.info(f"Compute CMS for: {entity['wd_label']} ({entity['wd_id']})")
                    entity["cms"] = self.calculate_cms(image_embeddings[entity["type"]], entity_embeddings)
                    entities_cms[entity["wd_id"]] = entity["cms"]

                if entity["cms"] > document_cms[entity["type"]]:
                    document_cms[entity["type"]] = entity["cms"]

                cms = document_cms

                unique_entities = set()
                logging.info("#### CMS for individual entities")
                list1 = []
                for entity in entities_cms:
                    logging.debug(entity)
                    if entity["wd_id"] not in unique_entities and entity["type"] in ["PERSON"]:
                        logging.info(f"{entity['type']} - {entity['wd_label']} ({entity['wd_id']}): {entity['cms']}")
                        unique_entities.add(entity["wd_id"])
                    entity_cms = entity['cms']

                    if entity['cms'] == None:
                        pass
                    elif entity_cms > 0.50:
                        abc = entity['wd_label']+':'+entity['cms']
                        list1.append(abc)
                temp[key] = list1

            except:
                    pass
            l = l + 1
            print("l=",l)
            logging.info("#### CMS for the whole document")
            logging.info(f"CMPS: {cms['PERSON']}")
            if len(temp)!= 0:
                constraints[k] = temp
            if l/l== 1:          
                print("l/2",l)
                with open("goodnews_constraints.json", "w") as fp:
                    json.dump(constraints , fp) 

    return 0


if __name__ == "__main__":
    sys.exit(main())
