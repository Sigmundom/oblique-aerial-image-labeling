import json
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from pycocotools.coco import COCO
import shapely.geometry as sg
from rasterio.features import rasterize
from pycocotools.mask import merge, decode, area, iou, frPyObjects


image_path = 'outputs/test/images/'

def evaluate_per_image(manual, auto):
    f = open('evaluation/stats.csv', 'w', encoding='utf-8')
    f.write('Image,IoU\n')
   
    for image_id, m_anns in manual.imgToAnns.items():
        image_data = manual.loadImgs(image_id)[0]
        image = plt.imread(image_path + image_data['file_name'])
        RLEs = [manual.annToRLE(ann) for ann in m_anns]
        m_RLE = merge(RLEs)
        m_mask = decode(m_RLE)

        a_ann_ids = auto.getAnnIds(image_id)
        a_ann = auto.loadAnns(a_ann_ids)[0]
        a_RLE = auto.annToRLE(a_ann)
        a_mask = decode(a_RLE)


        intersection = merge([a_RLE, m_RLE], intersect=True)
        union = merge([a_RLE, m_RLE])
        intersection_mask = decode(intersection)

        only_m = m_mask-intersection_mask
        only_a = a_mask-intersection_mask
        mask = intersection_mask + 2*only_m + 3*only_a

        a_total = area(union)
        a_intersection = area(intersection)
        # a_only_m = only_m.sum()
        # a_only_a = only_a.sum()

        iou = a_intersection/a_total
        f.write(f'{image_id},{iou}\n') 

        plt.imshow(image)
        plt.imshow(a_mask, alpha=0.4*(a_mask>0))
        plt.axis('off')
        plt.savefig(f'evaluation/auto_{image_id}.jpg', bbox_inches='tight', pad_inches = 0)
        plt.close('all')

        plt.imshow(image)
        plt.imshow(m_mask, alpha=0.4*(m_mask>0))
        plt.axis('off')
        plt.savefig(f'evaluation/manual_{image_id}.jpg', bbox_inches='tight', pad_inches = 0)
        plt.close('all')

        # plt.imshow(mask, alpha=0.3*(mask>0), cmap='jet')
        # plt.savefig(f'evaluation/img_{image_id}.jpg', bbox_inches='tight', pad_inches = 0)
        # plt.savefig(f'evaluation/img_{image_id}.jpg', bbox_inches='tight', pad_inches = 0)
        plt.imsave(f'evaluation/mask_{image_id}.png', mask)

        # exit()
    f.close()

colors = [ "red", "blue", "green", "yellow", "purple", "orange" ]

def evaluate_per_annotation(manual: COCO):
    auto = get_auto_annotations()
    print(len(auto))
    ious = np.empty_like(auto, dtype=np.float32)
    for annot in auto:
        image_id = annot['image_id']
        gt_annots = manual.imgToAnns[image_id]
        max_iou = 0
        dt = frPyObjects(annot['segmentation'], 768, 768)
        gt = [manual.annToRLE(gt_annot) for gt_annot in gt_annots]
        iscrowd = [x['iscrowd'] for x in gt_annots]
        cur_iou = iou(dt, gt, iscrowd)
        ious[annot['id']-1] = cur_iou.max()
    print(ious)
    print(np.mean(ious[ious!=0]))
    print(np.median(ious))
        # print(annot['id'], cur_iou.max())
        # for gt_annot in gt_annots:
        #     gt = [manual.annToRLE(gt_annot)]
        #     cur_iou = iou(dt, gt, [gt_annot['iscrowd']])
        #     print(cur_iou)
            # if cur_iou > max_iou:
            #     max_iou = cur_iou
        # annot['iou'] = max_iou
        # print(annot['id'], max_iou)
        # if max_iou < 0.5:
        #     print(annot)
    


    # for image_id, image_data in manual.imgs.items():
    #     image = plt.imread(image_path+image_data['file_name'])
    #     anns = list(filter(lambda ann: ann['image_id'] == image_id, auto))
        
    #     plt.imshow(image)
    #     ax = plt.gca()
        
    #     for i, ann in enumerate(anns):
    #         # xy = [[x, 1024-y] for x, y in ann['segmentation']]
    #         xy = ann['segmentation'][0]
    #         x = xy[::2]
    #         y = xy[1::2]
    #         xy = np.array([x, y]).T
    #         ax.add_patch(Polygon(xy, facecolor=colors[i%len(colors)], alpha=0.4))
        
    #     plt.axis('off')
    #     plt.savefig(f'evaluation/{image_id}.png', bbox_inches='tight', pad_inches = 0)
    #     plt.close()
    #     # print(image_id, image_data)
    

    # auto_masks = []

    # for ann in manual.anns.values():
    #     print(ann)


def get_auto_annotations():
    annotations = []
    back = COCO('outputs/back/annotations/instances_train.json')
    front = COCO('outputs/front/annotations/instances_train.json')
    left = COCO('outputs/left/annotations/instances_train.json')
    right = COCO('outputs/right/annotations/instances_train.json')

    annot_id = 1
    back_annots = back.loadAnns(back.getAnnIds([120, 79]))
    for annot in back_annots:
        annot['image_id'] = 5 if annot['image_id'] == 120 else 6
        annot['id'] = annot_id
        annot_id += 1
    annotations.extend(back_annots)

    front_annots = front.loadAnns(front.getAnnIds([121, 134]))
    for annot in front_annots:
        annot['image_id'] = 3 if annot['image_id'] == 121 else 4
        annot['id'] = annot_id
        annot_id += 1
    annotations.extend(front_annots)

    left_annots = left.loadAnns(left.getAnnIds([104, 148]))
    for annot in left_annots:
        annot['image_id'] = 7 if annot['image_id'] == 104 else 8
        annot['id'] = annot_id
        annot_id += 1
    annotations.extend(left_annots)

    right_annots = right.loadAnns(right.getAnnIds([62, 73]))
    for annot in right_annots:
        annot['image_id'] = 1 if annot['image_id'] == 62 else 2
        annot['id'] = annot_id
        annot_id += 1
    annotations.extend(right_annots)

    return annotations
    


if __name__ == '__main__':
    manual = COCO('outputs/test/annotations/manual.json')
    auto = COCO('outputs/test/annotations/segmentation.json')

    evaluate_per_image(manual, auto)
    # evaluate_per_annotation(manual)
    