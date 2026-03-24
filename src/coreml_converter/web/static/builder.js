function builder(baseModelData) {
    return {
        baseModel: baseModelData || {},
        loras: [],
        modelName: (baseModelData && baseModelData.name ? baseModelData.name + '-custom' : 'custom-model'),
        building: false,

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

        async startBuild() {
            if (!this.baseModel.id || this.building) return;
            this.building = true;

            const formData = new FormData();
            formData.append('base_model', JSON.stringify(this.baseModel));
            formData.append('loras', JSON.stringify(this.loras));
            formData.append('name', this.modelName);

            try {
                const resp = await fetch('/build/start', {
                    method: 'POST',
                    body: formData,
                });
                // The server returns a redirect to /build/{job_id}
                if (resp.redirected) {
                    window.location.href = resp.url;
                } else if (resp.ok) {
                    // If not redirected, try to get the location from response
                    const text = await resp.text();
                    window.location.href = resp.url;
                } else {
                    const errText = await resp.text();
                    alert('Build failed to start: ' + errText);
                    this.building = false;
                }
            } catch (err) {
                alert('Error: ' + err.message);
                this.building = false;
            }
        },
    };
}
