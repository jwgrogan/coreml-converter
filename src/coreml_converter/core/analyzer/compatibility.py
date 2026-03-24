from __future__ import annotations
from coreml_converter.core.models import (
    CompatibilityReport, Conflict, LoRAEntry, ModelInfo, RiskLevel, Severity,
)

def check_compatibility(base_model: ModelInfo, loras: list[LoRAEntry]) -> CompatibilityReport:
    conflicts: list[Conflict] = []
    arch_match = True
    for entry in loras:
        if entry.model.base_architecture != base_model.base_architecture:
            arch_match = False
            conflicts.append(Conflict(
                lora_a=entry.model.name, lora_b=base_model.name,
                reason=f"Architecture mismatch: LoRA is {entry.model.base_architecture.value}, base is {base_model.base_architecture.value}",
                severity=Severity.WARNING,
            ))
    count = len(loras)
    lora_count_warning = None
    if count >= 6:
        lora_count_warning = "6+ LoRAs: likely to produce artifacts, proceed at own risk"
    elif count >= 4:
        lora_count_warning = "4-5 LoRAs: quality may degrade"
    if not arch_match:
        risk = RiskLevel.HIGH
    elif count >= 6:
        risk = RiskLevel.HIGH
    elif count >= 4 or conflicts:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.LOW
    return CompatibilityReport(
        is_compatible=arch_match, architecture_match=arch_match, dimension_check=None,
        conflicts=conflicts, lora_count_warning=lora_count_warning, overall_risk=risk,
    )
