from soma.qt_gui.qt_backend import Qt
import anatomist.api as ana
import os
from soma import aims
from PIL import Image, ImageDraw, ImageFont


def create_grid(image_files, n_cols, out_path, title=None, subject_names=None):
    # load all images
    imgs = [Image.open(f) for f in image_files]
    # calculate max width and height
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    # calculate number of rows
    n_rows = (len(imgs) + n_cols - 1) // n_cols

    title_h = 0
    legend_h = 0
    font_size = 36
    if title or subject_names:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        if title:
            title_h = font.getbbox(title)[3] - font.getbbox(title)[1] + 10  # add margin
        if subject_names:
            legend_h = font.getbbox("Test")[3] - font.getbbox("Test")[1] + 15  # idem

    # create a new blank image with space for title and legend
    grid = Image.new('RGB', (n_cols * w, n_rows * (h + legend_h) + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Draw title
    if title:
        bbox = draw.textbbox((0, 0), title, font=font)  
        text_w = bbox[2] - bbox[0]
        x = (grid.width - text_w) // 2
        draw.text((x, 5), title, fill=(0, 0, 0), font=font)

    # Paste images and draw subject names
    for idx, im in enumerate(imgs):
        i, j = divmod(idx, n_cols)
        x0 = j * w
        y0 = title_h + i * (h + legend_h)
        grid.paste(im, (x0, y0))

        if subject_names:
            subj = subject_names[idx]
            text_bbox = draw.textbbox((0, 0), subj, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text((x0 + (w - text_w) // 2, y0 + h-50), subj, fill=(0, 0, 0), font=font)

    grid.save(out_path)
    print(f"Snapshot of the block available at {out_path}")

# select sulcus name, hemisphere, and subjects list

base_dir = (
    '/neurospin/dico/rmenasria/Runs/03_main/Input/mnt/'
    'derivatives/soft-brainvisa_ses-baselineYear1Arm1_ver-5.2_morphologist'
)
SUBJECTS_TEST = ['sub-NDARINVZXPAWB32', 'sub-NDARINV0A6WVRZY']
HEMISPHERE = 'R'
#SULCUS_NAME = 'S.T.s._right'  # Trop haut dans la nomenclature, donc colorie trop de sillons
SULCUS_NAME = 'F.C.M.post._right'  

CAMERA_PARAMS_FCM = {
    'view_quaternion': [-0.26836308836937, -0.323044091463089, -0.315022945404053, -0.851107776165009],
    'zoom': 1.25,
}
CAMERA_PARAMS_STS = {
    'view_quaternion': [-0.506462574005127, 0.718799889087677, 0.241088211536407, -0.410729736089706],
    'zoom': 1.25,
}

CAMERA_PARAMS_DEFAULT = {
    'view_quaternion': [-0.491121023893356, 0.638469696044922, 0.236046299338341, -0.543542921543121],
    'zoom': 1.25,
}

SUBJECTS_PREMA_CLASSIF_27_32_ABCD_STS= [
    "sub-NDARINV5HH9FDM1",
    "sub-NDARINVN1TKYKW1",
    "sub-NDARINVNBU3UA23",
    "sub-NDARINVDE17JJG5",
    "sub-NDARINVE75NTPTJ",
    "sub-NDARINVJ5VHD38X",
    "sub-NDARINVPPN58EEF",
    "sub-NDARINVM6EBZ88E",
    "sub-NDARINVF41DEHAC",
    "sub-NDARINVTUUTV3B2",
    "sub-NDARINVCCPGPFR1",
    "sub-NDARINVKWBVX0UU",
]

SUBJECTS_PREMA_CLASSIF_27_32_ABCD_FCM = [
    "sub-NDARINV88JX7RKX",
    "sub-NDARINVEWDP96RH",
    "sub-NDARINVGBUMHM8K",
    "sub-NDARINV0GPKYMDC",
    "sub-NDARINVH68T1XNB",
    "sub-NDARINVR28A45T6",
    "sub-NDARINV4YPJW3P2",
    "sub-NDARINVWZRF2GE6",
    "sub-NDARINVCTVENUN4",
    "sub-NDARINVL9NUBDAN",
    "sub-NDARINV0RA4PBPV",
    "sub-NDARINV8KX9UXKE",
]

SUBJECTS_FULLTERMS_CLASSIF_27_32_ABCD_STS = [
    "sub-NDARINVPCCWYX33",
    "sub-NDARINV3BL9Z315",
    "sub-NDARINV1P29RX5F",
    "sub-NDARINVM68WD1CV",
    "sub-NDARINVWPTBB9N7",
    "sub-NDARINV749XW1TD",
    "sub-NDARINVANH9WRCV",
    "sub-NDARINVUN2A7AA5",
    "sub-NDARINV5GB3TA25",
    "sub-NDARINVZR16R6Y3",
    "sub-NDARINVHE58CENN",
    "sub-NDARINV5XJMU584",
]

SUBJECTS_FULLTERMS_CLASSIF_27_32_ABCD_FCM = [
    "sub-NDARINV7XL78LUT",
    "sub-NDARINVE3F19GUV",
    "sub-NDARINV9R3YVGD6",
    "sub-NDARINV5VGKMHCR",
    "sub-NDARINVVUMNAHPB",
    "sub-NDARINVZCDF5310",
    "sub-NDARINVDZLD38UM",
    "sub-NDARINVLZ97FPMY",
    "sub-NDARINV5HNAA4NT",
    "sub-NDARINVUDEPUVAK",
    "sub-NDARINVDUJJ740F",
    "sub-NDARINVLABWKL63",
]


SUBJECTS_PREMA_CLASSIF_27_ABCD_STS = [
    "sub-NDARINVTUUTV3B2",
    "sub-NDARINVV6XHZB59",
    "sub-NDARINVN0XCHU07",
    "sub-NDARINVA4V1CUK7",
    "sub-NDARINVJX8BERDK",
    "sub-NDARINVYZZDEXDF",
    "sub-NDARINV9UL0YHV1",
    "sub-NDARINVNMC8EYAA",
    "sub-NDARINVLFX8PTZ6",
    "sub-NDARINV55285WTD",
    "sub-NDARINVUTBRXV0G",
    "sub-NDARINV14BPGL15",
]


SUBJECTS_PREMA_CLASSIF_27_ABCD_FCM = [
    "sub-NDARINVH68T1XNB",
    "sub-NDARINVNNN5FFAL",
    "sub-NDARINV1KP8VFVR",
    "sub-NDARINV9ZW7Y2R7",
    "sub-NDARINVTZR5RRWU",
    "sub-NDARINVP3B4C2JE",
    "sub-NDARINVYX7ADDMR",
    "sub-NDARINVB1CK69PR",
    "sub-NDARINVVKCT9VLF",
    "sub-NDARINVVD3KF4HE",
    "sub-NDARINVKNBJ47HV",
    "sub-NDARINVFA0DV854",
]


SUBJECTS_FULLTERMS_CLASSIF_27_ABCD_STS = [
    "sub-NDARINV9TZYNDC7",
    "sub-NDARINVBYARCL2L",
    "sub-NDARINVJ46KXBP8",
    "sub-NDARINVDYACTVYF",
    "sub-NDARINVGM7YJH08",
    "sub-NDARINVBZZGY7AM",
    "sub-NDARINVJ8LA1XLG",
    "sub-NDARINVDL871HVM",
    "sub-NDARINV2CVWTNF7",
    "sub-NDARINVM68WD1CV",
    "sub-NDARINVK6W4APWD",
    "sub-NDARINVLMME1BK2",
]


SUBJECTS_FULLTERMS_CLASSIF_27_ABCD_FCM = [
    "sub-NDARINVZJV3HZEF",
    "sub-NDARINVJ51UCKWX",
    "sub-NDARINVLGZ21X0T",
    "sub-NDARINV6R64U2H8",
    "sub-NDARINV8TLTP9PT",
    "sub-NDARINVK2KFJDG5",
    "sub-NDARINVV1LRPT07",
    "sub-NDARINVGGDFBWCL",
    "sub-NDARINVKTEZYKCL",
    "sub-NDARINVRR22WRZJ",
    "sub-NDARINVKZDAJBD7",
    "sub-NDARINVTY7NL56R",
]


# here select the list 
SUBJECTS = SUBJECTS_PREMA_CLASSIF_27_ABCD_FCM



app = Qt.QApplication.instance() or Qt.QApplication([])
a = ana.Anatomist()
windows = []
w = a.createWindow('3D')
windows.append(w)
w.assignReferential(a.centralRef)

if SULCUS_NAME == 'F.C.M.post._right':
        CAMERA_PARAMS = CAMERA_PARAMS_FCM
elif SULCUS_NAME == 'S.T.s._right':
    CAMERA_PARAMS = CAMERA_PARAMS_STS
else:
    CAMERA_PARAMS = CAMERA_PARAMS_DEFAULT 
    
w.camera(
    zoom=CAMERA_PARAMS['zoom'],
    view_quaternion=CAMERA_PARAMS['view_quaternion'],
    force_redraw=True
)


for subj in SUBJECTS:

    da = os.path.join(
        base_dir, subj, 'ses-1', 'anat', 't1mri',
        'default_acquisition', 'default_analysis'
    )

    mesh_file = os.path.join(da, 'segmentation', 'mesh', f"{subj}_{HEMISPHERE}white.gii")

    graph_file = os.path.join(
        da, 'folds', '3.1', 'deepcnn_session_auto',
        f"{HEMISPHERE}{subj}_deepcnn_session_auto.arg"
    )

    mesh = a.loadObject(mesh_file, hidden=False)
    mesh.loadReferentialFromHeader()


    w.addObjects(mesh)
    mesh.setMaterial(a.Material(diffuse=[1, 1, 1, 0.5]))

    graph = a.loadObject(graph_file, hidden=False)
    graph.loadReferentialFromHeader()


    w.addObjects(graph, add_graph_nodes=True)


    hie_path = aims.carto.Paths.findResourceFile('nomenclature/hierarchy/sulcal_root_colors.hie')
    nomenclature = a.loadObject(hie_path)
    
    # Unset the automatic coloring of the graph
    a.execute(
        'GraphDisplayProperties',
        objects=[graph],
        nomenclature_property='plouf'  # does not matter, just to unset the automatic coloring
    )

    # Select from the nomenclature. However, this does not work as expected if the node your want to select as STs for instance has children nodes
    # a.execute(
    #     'SelectByHierarchy',
    #     object=graph,
    #     nomenclature=nomenclature,
    #     names=SULCUS_NAME
    # )
   

    # # Select the sulcus in the graph
    # sf = ana.cpp.SelectFactory.factory()
    # print(f"Selected sulci for {subj}: {sf.selected()}")
    # selected = list(sf.selected()[0])


    # # Set the material for the selected sulcus
    # a.execute('SetMaterial', objects=selected, diffuse=[1.0, 0.35, 0.0, 1.0])
    # If you want to select the sulcus in the graph for Hierachical selection
    #a.execute('Select', unselect_objects=selected)



    # connect to the aim graph object to manually color the sulcus
    aims_graph = graph.graph()
    for vertex in aims_graph.vertices():
        label = vertex.get('label')
        if label == SULCUS_NAME:
            ana_vertex = vertex['ana_object']
            a.execute('SetMaterial', objects=[ana_vertex], diffuse=[1.0, 0.35, 0.0, 1.0])

    
  
    snapshot = True


    # set the camera position

    if snapshot:

        save_dir = "/neurospin/dico/rmenasria/Runs/03_main/Output/Figures/anat_snapshots/prema_27_ABCD"
        w.setHasCursor(0)
        fname = f"{subj}_{SULCUS_NAME}.png"
        img_path = os.path.join(save_dir, fname)
        w.snapshot(img_path, width=1200, height=900)

        # Close the window
        #w.close()


    #Remove objects from the window
    w.removeObjects([mesh, graph, nomenclature])

grid_dir = "/neurospin/dico/rmenasria/Runs/03_main/Output/Figures/anat_snapshots/prema_27_ABCD"
create_grid(
    image_files=[
        os.path.join(grid_dir, f"{subj}_{SULCUS_NAME}.png") for subj in SUBJECTS
    ],
    subject_names=SUBJECTS,
    n_cols= 4,
    out_path=os.path.join(
        grid_dir, f"{SULCUS_NAME}_grid.png"),
    title=f"{SULCUS_NAME}"
)


# Uncomment the following lines if you want to print camera infos periodically in the console. It allows to set manually the camera position in the code above.
# from PyQt5.QtCore import QTimer

# def print_camera_infos():
#     for idx, win in enumerate(windows):
#         try:
#             info = win.getInfos()
#             quat = info.get('view_quaternion', None)
#             zoom = info.get('zoom', None)
#             print(f"Fenêtre {idx} :")
#             print(f"  Quaternion : {quat}")
#             print(f"  Zoom       : {zoom}")
#         except Exception as e:
#             print(f"Erreur pour la fenêtre {idx} : {e}")

# timer = QTimer()
# timer.timeout.connect(print_camera_infos)
# timer.start(5000)



app.exec_()