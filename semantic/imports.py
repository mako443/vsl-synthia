import numpy as np
import os
import cv2

CLASS_IDS={'Building': 2, 'Road': 3, 'Sidewalk': 4, 'Fence': 5, 'Vegetation': 6, 'Pole': 7, 'Traffic Sign': 9, 'Traffic Light': 15}
CLASS_COLORS={'Building': (128,0,0), 'Road': (128,64,128), 'Sidewalk': (0,0,192), 'Fence': (64,64,128), 'Vegetation': (128,128,0), 'Pole': (192,192,128), 'Traffic Sign': (192,128,128), 'Traffic Light': (0,128,128)}

CAMERA_I=np.array([
    532.740352,     0,          640,
    0,              532.740352, 380,
    0,              0,          1
]).reshape((3,3))
CAMERA_I_INV=np.linalg.inv(CAMERA_I)

RELATIONSHIP_TYPES=('left','right','below','above','infront','behind')
DIRECTIONS=('Omni_F', 'Omni_B', 'Omni_R', 'Omni_L')

CORNER_NAMES=('top-left','top-right','bottom-left','bottom-right','center')
CORNERS=np.array(( (0.2, 0.2), (0.8,0.2), (0.2,0.8), (0.8,0.8), (0.5,0.5) )).reshape((5,2)) #Corners as relative (x,y) positions

'''
Colors in 8 clusters, RGB, summer+dawn+winter combined
'''
COLORS=np.array([[   29.6270532 ,  33.86763058,  44.62200417],
                    [134.37045104, 141.2597133 , 146.89439693],
                    [ 68.59381666,  78.86635468,  89.16569089],
                    [ 15.47688686,  17.52295666,  26.45806139],
                    [150.34269489, 172.11904894, 194.64450904],
                    [ 47.95475968,  54.83951892,  63.30084586],
                    [ 96.39856265, 107.18070884, 123.50450738],
                    [181.32395848, 171.22627764, 156.67673565]])
COLOR_NAMES=('dark-red', 'bright-gray', 'dark-beige', 'black', 'bright-beige', 'medium-red', 'tan', 'light-blue')

IMAGE_WIDTH, IMAGE_HEIGHT= 1280,760

'''
^MClass         R       G       B       ID

^MVoid          0       0       0       0
^MSky             128   128     128     1
^MBuilding        128   0       0       2
^MRoad            128   64      128     3
^MSidewalk        0     0       192     4
^MFence           64    64      128     5
^MVegetation      128   128     0       6
^MPole            192   192     128     7
^MCar             64    0       128     8
^MTraffic Sign    192   128     128     9
^MPedestrian      64    64      0       10
^MBicycle         0     128     192     11
^MLanemarking   0       172     0       12
^MReserved      -       -       -       13
^MReserved      -       -       -       14
^MTraffic Light 0       128     128     15
'''

class ViewObject:
    __slots__ = ['label', 'bbox', 'centroid_i', 'centroid_c', 'color']
    BG_DIST_THRESH=15

    def __init__(self, label, bbox, centroid_i, centroid_c, color):
        self.label=label
        self.bbox= np.array(bbox) if bbox is not None else None # (xmin,ymin,zmin,xmax,ymax,zmax)
        self.centroid_i= centroid_i
        self.centroid_c= centroid_c
        self.color=color #CARE: given as BGR

    def draw_on_image(self,img, draw_label=False):
        color=CLASS_COLORS[self.label]
        color=(color[2],color[1],color[0])
        if self.bbox is not None:
            cv2.rectangle(img, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[3]), int(self.bbox[4])), color, thickness=5)
        if draw_label:
            cv2.putText(img,self.label,(int(self.centroid_i[0])-50,int(self.centroid_i[1])),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=2)    
        else:
            cv2.circle(img, (int(self.centroid_i[0]),int(self.centroid_i[1])), 9, color, -1 )

    def __str__(self):
        return f'{self.label} at {self.centroid_i} color {self.color} dist {self.centroid_c[2]}'

    @classmethod
    def from_description_object(cls, do):
        dist = ViewObject.BG_DIST_THRESH*0.5 if do.distance=='foreground' else ViewObject.BG_DIST_THRESH*2.0
        centroid_i= np.hstack(( CORNERS[CORNER_NAMES.index(do.corner)]*(IMAGE_WIDTH,IMAGE_HEIGHT),dist ))
        color= COLORS[COLOR_NAMES.index(do.color)]
        return ViewObject(do.label, None, centroid_i, None, color)

def draw_scenegraph_on_image(img, scene_graph_or_relationships):
    relationships = scene_graph_or_relationships if type(scene_graph_or_relationships) is list else scene_graph_or_relationships.relationships
    
    for rel in relationships:
        if rel is None: continue
        for p in (rel[0], rel[2]):
            p.draw_on_image(img)

        p0,p1= (int(rel[0].centroid_i[0]), int(rel[0].centroid_i[1])), (int(rel[2].centroid_i[0]), int(rel[2].centroid_i[1]))
        cv2.arrowedLine(img, (p0[0],p0[1]), (p1[0],p1[1]), (0,0,255), thickness=4)
        cv2.putText(img,rel[1]+" of",(p0[0],p0[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=2)    

'''
DEPRECATED: SceneGraph should now be a collection of SGOs, no relationships anymore.
'''
class SceneGraph:
    def __init__(self):
        self.relationships=[]

    #Store relationship as (SG-object, rel_type, SG-object) triplet
    def add_relationship(self, sub, rel_type, obj):
        self.relationships.append( (sub,rel_type,obj) ) 

    #Also describes the objects corner and allows dublicates, making it equivalent to the Scene-Graph relationships used in Geometric Learning.
    def get_text_extensive(self):
        if len(self.relationships)==0: #TODO/CARE: this ok?
            return 'There is nothing to describe.'

        text=''
        for rel in self.relationships:
            sub, rel_type, obj=rel
            rel_text=f'In the {sub.corner} there is a {sub.color} {sub.label} that is {rel_type} of a {obj.color} {obj.label} in the {obj.corner}. '
           
            text+=rel_text

        return text   

    def is_empty(self):
        return len(self.relationships)==0     

    def draw_on_image(self,img):
        if self.is_empty(): return
        assert type(self.relationships[0][0])==ViewObject #only for debug-SGs
        draw_scenegraph_on_image(img, self)

        # for rel in self.relationships:
        #     for p in (rel[0], rel[2]):
        #         p.draw_on_image(img)

        #     p0,p1= (int(rel[0].centroid_i[0]), int(rel[0].centroid_i[1])), (int(rel[2].centroid_i[0]), int(rel[2].centroid_i[1]))
        #     cv2.arrowedLine(img, (p0[0],p0[1]), (p1[0],p1[1]), (0,0,255), thickness=3)
        #     cv2.putText(img,rel[1]+" of",(p0[0],p0[1]),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), thickness=2)

'''
DEPRECATED: DescriptionObject should be moved here
'''
class SceneGraphObject:
    __slots__ = ['label', 'color', 'corner']

    @classmethod
    def from_viewobject(cls, v):
        sgo=SceneGraphObject()
        sgo.label=v.label

        corner_distances= np.linalg.norm( CORNERS - (v.centroid_i[0:2]/(IMAGE_WIDTH,IMAGE_HEIGHT)), axis=1)
        sgo.corner= CORNER_NAMES[np.argmin(corner_distances)]

        color_distances= np.linalg.norm( COLORS-v.color, axis=1 )
        sgo.color=COLOR_NAMES[ np.argmin(color_distances) ]

        return sgo
    

    def __str__(self):
        return f'{self.color} {self.label} at {self.corner}'

    def get_text(self):
        return str(self)     

###
# New strategy: object descriptions
###
class DescriptionObject:
    __slots__= ['label','color', 'corner','distance']

    @classmethod
    def generate_caption(cls, description_objects):
        if len(description_objects)==0:
            return 'Empty graph.'

        text=""
        for do in description_objects:
            t=f'In the {do.corner} {do.distance} there is a {do.color} {do.label}. '
            text+=t
        return text

    @classmethod
    def from_viewobject(cls, v):
        do=DescriptionObject()
        do.label=v.label

        corner_distances= np.linalg.norm( CORNERS - (v.centroid_i[0:2]/(IMAGE_WIDTH,IMAGE_HEIGHT)), axis=1)
        do.corner= CORNER_NAMES[np.argmin(corner_distances)]

        color_distances= np.linalg.norm( COLORS-v.color, axis=1 )
        do.color=COLOR_NAMES[ np.argmin(color_distances) ]

        do.distance= 'foreground' if v.centroid_c[2]<ViewObject.BG_DIST_THRESH else 'background'

        return do   

    def __str__(self):
        return f'{self.color} {self.label} at {self.corner} {self.distance}'

    def get_text(self):
        return str(self)                 

if __name__=='__main__':
    point_c=np.array((0,0,2),dtype=np.float32)
    point_c[0:2]/=point_c[2]
    point_i=CAMERA_I@point_c
    print(point_i)