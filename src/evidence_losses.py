"""Response KD and static masked-score KD share the same views and scale."""
import torch
from torch.nn import functional as F


def evidence_loss(clean_student, masked_student, clean_teacher, masked_teacher,
                  objective="response"):
    with torch.autocast(device_type=clean_student.device.type, enabled=False):
        if objective == "response":
            prediction = clean_student.float() - masked_student.float()
            target = clean_teacher.detach().float() - masked_teacher.detach().float()
        elif objective == "masked":
            prediction = masked_student.float()
            target = masked_teacher.detach().float()
        else:
            raise ValueError(f"Unknown evidence objective: {objective}")
        # Huber delta=1 and mean over samples/classes. No hidden temperature
        # or scale matching: teacher/student response RMS are logged separately.
        return F.huber_loss(prediction, target, delta=1.0)
