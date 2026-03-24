function builder(baseModelData) {
    return {
        baseModel: baseModelData || {},
        loras: [],
        modelName: (baseModelData && baseModelData.name ? baseModelData.name + '-custom' : 'custom-model'),
        addLora(model, recommendedWeight, weightSource) {
            if (this.loras.find(l => l.model.id === model.id)) return;
            this.loras.push({
                model: model,
                weight: recommendedWeight || 1.0,
                recommended_weight: recommendedWeight,
                weight_source: weightSource,
            });
        },
        removeLora(index) {
            this.loras.splice(index, 1);
        },
        startBuild() {
            this.$refs.buildForm.submit();
        },
    };
}
