import argparse
import logging
import sys
import yaml
import json

from utils import allowed_image_file
from newsanalyzer import NewsAnalyzer
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

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
  
    cms = {"PERSON": 0.0}
    constraints ={}
    l=0
    im_cap = '../news_images/data/nytimes_cap_art1.json'          
    with open('nytimes_constraints.json', "r+") as fp:
      nytimes_cons= json.load(fp)
    with open(im_cap) as fp:
      im_name_cap= json.load(fp)
    # len(im_name_cap)
    keys_list = []
    for k,v in im_name_cap.items():
      if k not in nytimes_cons.keys():
        temp = {}
        for key in v['images'].keys():
          image = '../datasets/nytimes/images_processed/'+key+'.jpg'
          text = v['article_text']

          # check image input
          if not allowed_image_file(image):
            logging.error("Image extension unknown. Exiting ...")
            return 0
          keys_list.append(k) 
          try:
            cms, entities_cms = na.get_entity_cms(image_file=image, text=text, language=args.language)
            unique_entities = set()
            logging.info("#### CMS for individual entities")
            list1 = []
            list2 = []
            temp1 ={}
            for entity in entities_cms:
              logging.debug(entity)
              if entity["wd_id"] not in unique_entities and entity["type"] in ["PERSON"]:
                logging.info(f"{entity['type']} - {entity['wd_label']} ({entity['wd_id']}): {entity['cms']}")
                unique_entities.add(entity["wd_id"])
              entity_cms = entity['cms']
            
              if entity['cms'] == None: 
                pass
              elif entity_cms > 0.45:
                list1.append(entity['wd_label'])
                list2.append(entity['cms'])
                
            print("len(list1)=",len(list1))
            if len(list1)!= 0:    
              temp1['labels'] = list1
              temp1['confidences'] = list2
              temp[key] = temp1
            
          except:
            print("Oops!", sys.exc_info()[0], "occurred.")
        l = l + 1
        print("l=",l)
        logging.info("#### CMS for the whole document")
        logging.info(f"CMPS: {cms['PERSON']}")
        constraints[k] = temp 
        if l%10== 0: 
          nytimes_cons.update(constraints)      
          with open("nytimes_constraints.json", "w") as fp:
              json.dump(nytimes_cons , fp) 
    return 0



if __name__ == "__main__":
    sys.exit(main())
