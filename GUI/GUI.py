from MainWin import Ui_Dialog
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
from PyQt5 import QtCore, QtGui, QtWidgets
from Segformer import segformer_mit_b0
from predict import Predict1
from datapre import get_nor_img
from torchvision import transforms as T
from PIL import Image as PilImage
from utils import koutu
import torch
from PyQt5.QtGui import QPixmap

rootDir = "C:/Users/29779/Desktop/DL"


class MyDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.image_path = None

        self.pushButton_2.clicked.connect(self.select_image)
        self.pushButton_3.clicked.connect(self.show_image_in_label2)
        self.pushButton.clicked.connect(self.download_image)

    def select_image(self):
        file_dialog = QFileDialog()
        default_directory = "../data/"  # 默认目录
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片", default_directory,
                                                   "Image Files (*.png *.jpg *.jpeg)")
        self.image_path = file_path
        # print(self.image_path)
        if file_path:
            pixmap = QtGui.QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(384, 384, QtCore.Qt.IgnoreAspectRatio)
            self.label.setPixmap(scaled_pixmap)

            # 计算标签内可显示区域的中心位置
            label_width = self.label.width()
            label_height = self.label.height()
            x_offset = (label_width - 384) // 2
            y_offset = (label_height - 384) // 2

            # 调整图片位置实现居中
            self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.label.setContentsMargins(x_offset, y_offset, x_offset, y_offset)

    def show_image_in_label2(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = 2
        model = segformer_mit_b0(in_channels=3, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load("C:/Users/29779/Desktop/DL/results/people_detect_last.pt"))

        imgs = get_nor_img(self.image_path)

        IMAGE_SIZE = (384, 384)

        resize_t = T.Compose([T.Resize(IMAGE_SIZE)])
        imgss = resize_t(PilImage.open(self.image_path))

        ## 预测
        all_pre = Predict1(model, imgs, device)

        img = koutu(imgss, all_pre)

        # 将数组转换为 PIL 图像
        pil_image = PilImage.fromarray(img.astype('uint8'))

        # 将 PIL 图像显示在 self.label_2 上
        # 将 PIL 图像转换为 QImage
        q_image = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        self.label_2.setPixmap(pixmap)

        # 计算标签内可显示区域的中心位置
        label_width = self.label_2.width()
        label_height = self.label_2.height()
        x_offset = (label_width - 384) // 2
        y_offset = (label_height - 384) // 2

        # 调整图片位置实现居中
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label_2.setContentsMargins(x_offset, y_offset, x_offset, y_offset)

    def download_image(self):
        pixmap = self.label_2.pixmap()
        if pixmap:
            try:
                pixmap.save("downloaded_image.jpg")
                QtWidgets.QMessageBox.information(self, "下载成功", "图片已成功下载！")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "下载出错", f"下载失败：{e}")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    dialog = MyDialog()
    dialog.show()
    app.exec_()
