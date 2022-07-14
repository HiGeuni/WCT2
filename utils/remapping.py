
import numpy as np

class SegReMapping:
    def __init__(self, mapping_name, min_ratio=0.02):
        self.label_mapping = np.load(mapping_name)
        self.min_ratio = min_ratio

    def cross_remapping(self, cont_seg, styl_seg, bprint=False):
        cont_label_info = []
        new_cont_label_info = []
        for label in np.unique(cont_seg):
            cont_label_info.append(label)
            new_cont_label_info.append(label)
        cont_label_info_before = cont_label_info
        style_label_info = []
        new_style_label_info = []
        for label in np.unique(styl_seg):
            style_label_info.append(label)
            new_style_label_info.append(label)
        style_label_info_before = style_label_info
        cont_set_diff = set(cont_label_info) - set(style_label_info)
        # Find the labels that are not covered by the style
        # Assign them to the best matched region in the style region
        for s in cont_set_diff:
            cont_label_index = cont_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in style_label_info:
                    new_cont_label_info[cont_label_index] = new_label
                    break

        new_cont_seg = cont_seg.copy()
        for i,current_label in enumerate(cont_label_info):
            new_cont_seg[(cont_seg == current_label)] = new_cont_label_info[i]

        n_pixels = new_cont_seg.shape[0] * new_cont_seg.shape[1]
        new_cont_ratio_info = []
        for label in new_cont_label_info:
            new_cont_ratio_info.append(np.sum(np.float32((new_cont_seg == label))[:])/n_pixels)
    
        cont_label_info = []
        for label in np.unique(new_cont_seg):
            cont_label_info.append(label)
        styl_set_diff = set(style_label_info) - set(cont_label_info)
        valid_styl_set = set(style_label_info) - set(styl_set_diff)
        for s in styl_set_diff:
            style_label_index = style_label_info.index(s)
            for j in range(self.label_mapping.shape[0]):
                new_label = self.label_mapping[j, s]
                if new_label in valid_styl_set:
                    new_style_label_info[style_label_index] = new_label
                    break
        new_styl_seg = styl_seg.copy()
        for i,current_label in enumerate(style_label_info):
            # print("%d -> %d" %(current_label,new_style_label_info[i]))
            new_styl_seg[(styl_seg == current_label)] = new_style_label_info[i]

        n_pixels = new_styl_seg.shape[0] * new_styl_seg.shape[1]

        new_style_ratio_info = []
        for label in new_style_label_info:
            new_style_ratio_info.append(np.sum(np.float32((new_styl_seg == label))[:])/n_pixels)
        return new_cont_seg, new_styl_seg
    
    def self_remapping(self, seg, bprint = False):
        if bprint:
            print("### Self Remapping !! ###")
        init_ratio = self.min_ratio
        # Assign label with small portions to label with large portion
        new_seg = seg.copy()
        [h,w] = new_seg.shape
        n_pixels = h*w
        # First scan through what are the available labels and their sizes
        label_info = []
        ratio_info = []
        new_label_info = []
        for label in np.unique(seg):
            ratio = np.sum(np.float32((seg == label))[:])/n_pixels
            label_info.append(label)
            new_label_info.append(label)
            ratio_info.append(ratio)
        for i,current_label in enumerate(label_info):
            if ratio_info[i] < init_ratio:
                for j in range(self.label_mapping.shape[0]):
                    new_label = self.label_mapping[j,current_label]
                    if new_label in label_info:
                        index = label_info.index(new_label)
                        if index >= 0:
                            if ratio_info[index] >= init_ratio:
                                ratio_info[index] += ratio_info[i]
                                ratio_info[i] = 0
                                new_label_info[i] = new_label
                                break
        for i,current_label in enumerate(label_info):
            new_seg[(seg == current_label)] = new_label_info[i]
        idx = 0
        while idx < len(new_label_info):
            if ratio_info[idx] == 0:
                del new_label_info[idx]
                del ratio_info[idx]
            else:
                idx += 1
        
        return new_seg
