# PCF-Grasp
This repo will be update soon!

## PCF-Grasp: Pointclouds Completion of Feature for 6-DoF Grasp

[paper](), [video-bilibili](), [video-youtube](https://www.youtube.com/watch?v=w59oeQmOeNQ)

## Citation
If you find our work useful, please cite.
```latex
```

## Download Model and Dataset

### Prepare code and file

```bash
#only for test
git clone https://github.com/ChengYaofeng/PCF-Grasp.git
cd PCF-Grasp

└── PCF-Grasp
    └── pcfgrasp_method

#for train
git clone https://github.com/ChengYaofeng/PCF-Grasp.git
cd PCF-Grasp
mkdir acronym

└── PCF-Grasp
    ├── acronym
    └── pcfgrasp_method
```

### Installation

Create the conda env. CUDA 11.1, Pytorch 1.8

```bash
conda env create -f pcf.yaml
```

If you want to pretrain, use follow to setup CD.

```bash
conda activate pcf
cd extensions/chamfer_distance
python setup.py install
```

### Model

Download our trained models from [baidu cloud disk](), Extract code:.

### Dataset

Our dataset followed the [contact-graspnet](https://github.com/NVlabs/contact_graspnet), but we only placed one object in each scene.

The acronym file should be created as [this](https://github.com/NVlabs/acronym#using-the-full-acronym-dataset). After this step, the acronym file should be the same as:

```bash
└── acronym
    ├── grasps
    └── meshes
```

Then, you can follow [contact-graspnet](https://github.com/NVlabs/contact_graspnet) create new scenes or just download our scenes [here](). Extract it to acronym as:

```bash
└── acronym
    ├── grasps
    ├── meshes
    ├── scene_contacts
    └── splits
```

## Running
### Train
We recommend the bach_size at least 5, it is because sometimes the virtual camera can't capture some object as the object and camera are randomly placed.
+ Pointclouds Completion
```bash
cd /PCF-Grasp/pcfgrasp_method
bash ./scripts/pretrain.sh
```

+ 6-DoF Grasp
```bash
cd /PCF-Grasp/pcfgrasp_method
bash ./scripts/train.sh
```
---
### Inference
+ grasp inference

```bash
cd /PCF-Grasp/pcfgrasp_method
bash ./scripts/inference.sh
```

+ point completion inference

```bash
cd /PCF-Grasp/pcfgrasp_method
bash ./scripts/pre_inference.sh
```

+ real world inference

We use realsense d435 camera in our code. If you are the same camera and want to test in the real world scenes, you can use our code directly. Download [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

```bash
------------download detectron2-------
cd /PCF-Grasp/pcfgrasp_method
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
-----------------run code-------------
bash ./scripts/real_world_inference.sh
```

What's more, if you want to test on robot. You can create a msg file as `objects_grasp_pose.msg` with follow code. It will publish rostopic as topicname '/grasp_pose'. You can use `grasp_pose = rospy.wait_for_message('/grasp_pose', objects_grasp_pose)` to recieve grasp pose.

```bash
int32[] obj_index
geometry_msgs/Pose[]  grasp_pose
```

## License
MIT-LICENSE
