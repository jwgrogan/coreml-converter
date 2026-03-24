function builder() {
    return {
        loras: [],
        addLora(model, recommendedWeight, weightSource) {
            if (this.loras.find(l => l.model.id === model.id)) return;
            this.loras.push({
                model: model,
                weight: recommendedWeight || 1.0,
                recommended_weight: recommendedWeight,
                weight_source: weightSource,
            });
        },
        removeLora(id) {
            this.loras = this.loras.filter(l => l.model.id !== id);
        },
    };
}
