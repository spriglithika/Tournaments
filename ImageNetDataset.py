import io
import os
from zipfile import ZipFile

from PIL import Image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, dataroot: str, train: bool = True, transform=None):
        self.zfpath = os.path.join(
            dataroot,
            f"{'train' if train else 'val'}_blurred.zip",
        )

        self.transform = transform  # ✅ Store the transform

        # Avoid reusing the file handle created here, for known issue with multi-worker:
        self.zf = None
        with ZipFile(self.zfpath) as zf:
            self.imglist: list[str] = [
                path for path in zf.namelist()
                if path.endswith(".jpg")
            ]

        # Images are structured in directories based on class
        with open(os.path.join(dataroot, "map_clsloc.txt")) as f:
            def parse_row(row: str) -> tuple[str, int]:
                classname, classnum, _ = row.split()
                return classname, (int(classnum) - 1)
            self.classes: dict[str, int] = dict(parse_row(row) for row in f)

    def get_label(self, path: str) -> int:
        if not path.endswith(".jpg"):
            raise ValueError(f"Expected path to image, got {path}")
        classname: str = path.split("/")[-2]
        return self.classes[classname]

    def __len__(self):
        return len(self.imglist)
    def get_items(self):
        return [(imgpath, self.get_label(imgpath)) for imgpath in self.imglist]

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        if self.zf is None:
            self.zf = ZipFile(self.zfpath)

        imgpath = self.imglist[idx]
        img = Image.open(io.BytesIO(self.zf.read(imgpath))).convert("RGB")
        label = self.get_label(imgpath)

        # if self.transform:x
            # img = self.transform(img)  # ✅ Apply the transform

        return img, label