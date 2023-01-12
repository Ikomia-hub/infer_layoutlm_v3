# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from infer_layoutlm_v3.utils import polygon2bbox, normalize_box, unnormalize_box, iob_to_label
import copy
from ikomia import core, dataprocess
from transformers import AutoModelForTokenClassification, AutoProcessor
import torch
import numpy as np
from ikomia.utils import strtobool
import random

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferLayoutlmV3Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.cuda = torch.cuda.is_available()
        self.model_name = "jinhybr/OCR-LayoutLMv3-Invoice"
        self.checkpoint_path = ""
        self.checkpoint = False
        self.update = False
        
    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.cuda = strtobool(param_map["cuda"])
        self.model_name = str(param_map["model_name"])
        self.pretrained = strtobool(param_map["checkpoint"])
        self.checkpoint_path = param_map["checkpoint_path"]
        self.update = strtobool(param_map["update"])

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["cuda"] = str(self.cuda)
        param_map["model_name"] = str(self.model_name)
        param_map["checkpoint"] = str(self.checkpoint)
        param_map["checkpoint_path"] = self.checkpoint_path
        param_map["update"] = str(self.update)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferLayoutlmV3(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)

        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric output
        self.addOutput(dataprocess.CNumericIO())
        # Add input
        self.addInput(dataprocess.CNumericIO())
    
        self.model = None
        self.processor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.class_names = None
        self.colors = None
        # Create parameters class
        if param is None:
            self.setParam(InferLayoutlmV3Param())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, image, boxes_norm, words,  height, width):

        # Get parameters values
        param = self.getParam()

        # Encoding data for the model. It first applies feature extraction (resize + normalization)
        # and then tokenization (wordpiece), with turns words and bounding boxes  in token-level 
        # "input_ids", "attention_mask", "token_type_ids", "bbox"
        encoding = self.processor(image, words, boxes=boxes_norm, return_tensors="pt")

        # Move tensors to the selected device
        for k,v in encoding.items():
            encoding[k] = v.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        logits = outputs.logits
        predictions = logits.argmax(-1).squeeze().tolist()

        # Get bounding boxes
        token_boxes = encoding.bbox.squeeze().tolist()
        true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]

        # Get tokens and boxes that are at the start of a given word, "true".
        true_predictions = [self.id2label[pred] for idx, pred in enumerate(predictions)]
        
        # Prepare graphics output
        for pred, box in zip(true_predictions, true_boxes):
            class_index = self.class_names.index(pred)
            # Box
            x_obj = float(box[0])
            y_obj = float(box[1])
            h_obj = (float(box[3]) - y_obj)
            w_obj = (float(box[2]) - x_obj)
            prop_rect = core.GraphicsRectProperty()
            prop_rect.pen_color = self.colors[class_index]
            graphics_box = self.graphics_output.addRectangle(x_obj, y_obj, w_obj, h_obj, prop_rect)

            # Label
            predicted_label = iob_to_label(pred).lower()
            prop_text = core.GraphicsTextProperty()
            prop_text.font_size = 10
            prop_text.color = self.colors[class_index]
            self.graphics_output.addText(predicted_label, box[0], box[1]-30, prop_text)


    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        param = self.getParam()
    
        # Get input image
        input = self.getInput(0)
        image = input.getImage()
        height, width,  _ = image.shape  
        # Get input graphics (boxes)
        graphics_input = self.getInput(1)

        # Check if there are boxes as input
        if graphics_input.isDataAvailable():
            polygons = graphics_input.getItems()
            boxes = []

        # Create batch of images containing text
        for polygon in polygons:
            pts = polygon.points
            pts = np.array([[pt.x, pt.y] for pt in pts])
            x, y, w, h = polygon2bbox(pts)
            boxes.append([x, y, w, h])

        # Turn coordinates into (left, top, left+width, top+height) format
        actual_boxes = []
        for x, y, w, h in boxes:
            actual_box = [x, y, x + w, y + h]
            actual_boxes.append(actual_box)

        # Word-level bounding boxes. Each bounding box should be normalized to be on a 0-1000 scale.
        boxes_norm = [normalize_box(box, width, height) for box in actual_boxes]

        # Get text input from OCR
        input_data = self.getInput(2)
        if input_data.isDataAvailable():
            data = input_data.getAllLabelList()

        list_words = [word for word in data[0] if "[" and "]" in word]
        list_words = [s.strip('[') for s in list_words]
        words = [s.strip(']') for s in list_words]

        # Get output :
        # Prepare outputs
        self.graphics_output = self.getOutput(1)
        self.graphics_output.setNewLayer("layoutlmv3")
        self.graphics_output.setImageIndex(0)

        if param.update or self.model is None:
            # Loading processor which is use to encode data for the model
            self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", 
                                                           apply_ocr=False
                                                           )

            # Loading fine-tuned model from the HUB or local checkpoint
            # LayoutLMv3 Model with a token classification head on top  
            # for sequence labeling (information extraction)
            # tasks such as FUNSD, SROIE, CORD and Kleister-NDA.
            model_id = None
            if param.checkpoint is False:
                model_id = param.model_name
                self.model = AutoModelForTokenClassification.from_pretrained(model_id)
            else:
                model_id = param.checkpoint_path

            self.model = AutoModelForTokenClassification.from_pretrained(model_id)
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.model.to(self.device)

            print("Will run on {}".format(self.device.type))
            # Get id and label name
            self.id2label = self.model.config.id2label
            self.class_names = list(self.model.config.id2label.values())

            # Color palette
            random.seed(42)
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
            param.update = False

        # Inference
        self.infer(image, boxes_norm, words, height, width)

        # Step progress bar:
        self.emitStepProgress()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferLayoutlmV3Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_layoutlm_v3"
        self.info.shortDescription = "Information extraction from document images using LayoutLMv3."
        self.info.description = "This plugin proposes inference for document layout analysis "\
                                "using LayoutlMv3. The main task of LayoutLMv3 is extraction "\
                                "of key information from image documents. "\
                                "Text and visual features are used as inputs."
        self.info.path = "Plugins/Python/Classification"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Yupan Huang and Tengchao Lv and Lei Cui and Yutong Lu and Furu Wei"
        self.info.article = "LayoutLMv3: Pre-training for Document AI with Unified"\
                            "Text and Image Masking"
        self.info.journal = "Association for Computing Machinery"
        self.info.year = 2022
        self.info.license = "Attribution-NonCommercial-ShareAlike 4.0"\
                            "International (CC BY-NC-SA 4.0)"
        # URL of documentation
        self.info.documentationLink = "https://arxiv.org/pdf/2204.08387.pdf"
        # Code source repository
        self.info.repository = "https://github.com/microsoft/unilm/blob/master/layoutlmv3/README.md"
        # Keywords used for search
        self.info.keywords = "LayoutLM, document ai, transformers, vision-and-language, Huggingface"

    def create(self, param=None):
        # Create process object
        return InferLayoutlmV3(self.info.name, param)
