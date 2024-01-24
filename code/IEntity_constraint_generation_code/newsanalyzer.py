import logging
import numpy as np
import os
import sys

from crawl_image import download_bing_images

from metrics import cossim

from named_entity_linking.nerd import link_annotations, fix_entity_types
from named_entity_linking.spacy_ner import get_spacy_annotations
from named_entity_linking.wikifier import get_wikifier_annotations

from utils import allowed_image_file

from visual_descriptors.person_embedding import FacialFeatureExtractor, agglomerative_clustering
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

_CUR_DIR = os.path.dirname(__file__)
_IMAGE_DIR = os.path.join(_CUR_DIR, "reference_images")


class NewsAnalyzer:
    def __init__(self, wikifier_key, config):
        person_model_dir = os.path.join(_CUR_DIR, config["models"]["person"])
        if not os.path.exists(person_model_dir):
            logging.error(f"Cannot find folder <{person_model_dir}>. Please follow the README for instructions.")
            sys.exit()

        logging.info("Building network for for person verification ...")
        self.FE_face = FacialFeatureExtractor(model_path=person_model_dir)
        self.config = config
        self.wikifier_key = wikifier_key

    def get_linked_entities(self, text, language):
        logging.info("Getting linked entities ...")
        spacy_annotations = get_spacy_annotations(text, language=language)
        logging.debug(spacy_annotations)

        wikifier_annotations = get_wikifier_annotations(text, language=language, wikifier_key=self.wikifier_key)
        logging.debug(wikifier_annotations)

        linked_entities = link_annotations(spacy_annotations, wikifier_annotations)
        linked_entities = fix_entity_types(linked_entities)
        logging.debug(linked_entities)

        return linked_entities

    def get_image_embedding(self, image_file, entity_type):
        if not allowed_image_file(image_file):
            logging.warning(f"{image_file}: Image extension unknown")
        # print("inside image embeddings")    

        if entity_type == "PERSON":
            return self.FE_face.get_img_embedding(image_file)
        else:
            return None           

        return None

    def get_entity_embeddings(self, entity):
        embeddings = []

        # get embeddings for images in download folder
        image_folder = os.path.join(_IMAGE_DIR, entity["wd_id"])
        for image_file in os.listdir(image_folder):
            if not allowed_image_file(image_file):
                continue

            image_path = os.path.join(_IMAGE_DIR, entity["wd_id"], image_file)
            logging.debug(f"Getting embedding for {image_path} ...")
            embedding = self.get_image_embedding(image_path, entity_type=entity["type"])
            if embedding:
                embeddings.extend(embedding)

        return embeddings

    def calculate_cms(self, image_embeddings, entity_embeddings):
        entity_sims = cossim(
            np.asarray(image_embeddings, dtype=np.float32), np.asarray(entity_embeddings, dtype=np.float32)
        )
        if self.config["operator"] == "max":
            return np.max(entity_sims)

        if self.config["operator"].startswith("q") and len(self.config["operator"]) > 1:  
            return np.quantile(entity_sims, float(self.config["operator"][1:]) / 100)

        logging.warning(f"Unknown operator {self.config['operator']}. Using max instead ...")
        return np.max(entity_sims)

    def get_entity_cms(self, image_file, text, language="en"):
        linked_entities1 = self.get_linked_entities(text, language)
        linked_entities = []
        for i in range(len(linked_entities1)):
        	if linked_entities1[i]['type'] == "PERSON":
        		linked_entities.append(linked_entities1[i])

        # get image embeddings from image
        logging.info("Extracting image features ...")
        image_embeddings = {}
        image_embeddings["PERSON"] = self.get_image_embedding(image_file, entity_type="PERSON")

        if len(image_embeddings['PERSON'])!=0:    
            document_cms = {"PERSON": 0.0}
            entities_cms = {}

            for entity in linked_entities:  # TODO: can be optimized for speed using multithreading


                # if an entity is mentioned multiple times, the result is already calculated
                if entity["wd_id"] in entities_cms:
                    entity["cms"] = entities_cms[entity["wd_id"]]
                    continue

                logging.info(f"Download reference images for: {entity['wd_label']} ({entity['wd_id']})")  #James P. Hoffa (Q711662)
                logging.debug(entity)

                # download reference images for linked entities and store in download folder
                download_bing_images(
                    entity=entity["wd_label"],
                    entity_type=entity["type"],
                    num_images=self.config["num_reference_images"],
                    img_license="all",
                    download_folder=os.path.join(_IMAGE_DIR, entity["wd_id"]),
                )

                # get image embeddings for reference images
                logging.info(f"Extract image embeddings for: {entity['wd_label']} ({entity['wd_id']})")  #James P. Hoffa (Q711662)
                entity_embeddings = self.get_entity_embeddings(entity)

                if len(entity_embeddings) != 0:               	
                    if self.config["face_clustering"] and entity["type"] == "PERSON":
                        entity_embeddings = agglomerative_clustering(entity_embeddings)
                    logging.info(f"Compute CMS for: {entity['wd_label']} ({entity['wd_id']})")
                    entity["cms"] = self.calculate_cms(image_embeddings[entity["type"]], entity_embeddings)
                    entities_cms[entity["wd_id"]] = entity["cms"]  


                if entity["cms"] is not None and entity["cms"] > document_cms[entity["type"]]:
                    document_cms[entity["type"]] = entity["cms"]
                else:
                    pass    

            return document_cms, linked_entities
