import layoutparser as lp
from google.cloud import vision
import pandas as pd
import numpy as np
import cv2

api_key_path = r"vision-ai-api-413019-4716d3c323e9.json"

#image1 = cv2.imread("")

model_path = "layout_parser_weights\model_final.pth"

def prediction(image_path,model_path):
    ocr_agent = lp.GCVAgent.with_credential(api_key_path,languages = ['en'])
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                    model_path,
                                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.95],
                                    label_map={0:"HIC_table",1: "US_table", 2: "None", 3: "None", 4:"None", 5:"None"})


    f = open("logging.txt","w")
    image = cv2.imread(image_path)
    f.write(f"prediction: {image.shape}")
    layout = model.detect(image)
    f.write(f"layout: {layout}")

    text_blocks = lp.Layout([b for b in layout if b.type=="US_table"])
    f.write(f"text_blocks: {text_blocks}")


    figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])


    text_blocks = lp.Layout([b for b in text_blocks \
                    if not any(b.is_in(b_fig) for b_fig in figure_blocks)])



    h, w = image.shape[:2]

    left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(image)

    left_blocks = text_blocks.filter_by(left_interval, center=True)
    left_blocks.sort(key = lambda b:b.coordinates[1])

    right_blocks = [b for b in text_blocks if b not in left_blocks]
    right_blocks.sort(key = lambda b:b.coordinates[1])

    # And finally combine the two list and add the index
    # according to the order
    text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])
    
    f.write(f"text_blocks1: {text_blocks}")


    for block in text_blocks:
        segment_image = (block
                        .pad(left=10, right=10, top=10, bottom=10)
                        .crop_image(image))
            # add padding in each image segment can help
            # improve robustness
        #img = cv2.imread(segment_image)
        #f.write(f"segment_image: {img.shape}")
        layout = ocr_agent.detect(segment_image)
        block.set(text=layout, inplace=True)
        
        
        
    #filter_product_no = layout.filter_by(
    #lp.Rectangle(x_1=27, y_1=213, x_2=201, y_2=390),
    #soft_margin = {"left":10, "right":20}
    #)
    #lp.draw_text(segment_image, filter_product_no, font_size=10,with_box_on_text=True,text_box_width=6)


    filter_product_no = layout.filter_by(
        lp.Rectangle(x_1=27, y_1=213, x_2=201, y_2=390),
        soft_margin = {"left":10, "right":30,"top":10,"bottom": 20}
    )
    filter_product_no.get_texts()
    f.write(f"filter_product_no: {filter_product_no.get_texts()}")

    filter_apparatus_no = layout.filter_by(
        lp.Rectangle(x_1=565, y_1=213, x_2=574, y_2=390),
        soft_margin = {"left":10, "right":30,"top":10,"bottom": 20}
    )
    filter_apparatus_no.get_texts()
    
    f.write(f"filter_apparatus_no: {filter_apparatus_no.get_texts()}")

    #lp.draw_text(segment_image, filter_apparatus_no, font_size=10,with_box_on_text=True,text_box_width=2)


    filter_transducer_no = layout.filter_by(
        lp.Rectangle(x_1=983, y_1=214, x_2=995, y_2=389),
        soft_margin = {"left":10, "right":30,"top":10,"bottom": 20}
    )
    filter_transducer_no.get_texts()
    f.write(f"filter_transducer_no: {filter_transducer_no.get_texts()}")

    #lp.draw_text(segment_image, filter_transducer_no, font_size=10,with_box_on_text=True,text_box_width=5)



    table_dict = {
        "appartus_no":filter_apparatus_no.get_texts(),
        "product_no": filter_product_no.get_texts(),
        "transducer_no": filter_transducer_no.get_texts()
    }
    
    
    df = pd.DataFrame(table_dict)
    f.write(f"dataframe: {df.shape}")
    f.close()
    return df