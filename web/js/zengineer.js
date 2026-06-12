import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Shows the enhanced prompt(s) on Z-Engineer nodes after execution.

const PREVIEW_NODES = new Set(["ZEngineerEnhance", "ZEngineer"]);
const PREVIEW_PREFIX = "preview_";

app.registerExtension({
	name: "zengineer.PromptPreview",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!PREVIEW_NODES.has(nodeData.name)) {
			return;
		}

		function removePreviews() {
			if (!this.widgets) {
				return;
			}
			for (let i = this.widgets.length - 1; i >= 0; i--) {
				if (this.widgets[i].name?.startsWith(PREVIEW_PREFIX)) {
					this.widgets[i].onRemove?.();
					this.widgets.splice(i, 1);
				}
			}
		}

		function populate(text) {
			removePreviews.call(this);

			let values = text;
			if (!(values instanceof Array)) {
				values = [values];
			}
			for (const value of values) {
				if (value === undefined || value === null) {
					continue;
				}
				const w = ComfyWidgets["STRING"](
					this,
					PREVIEW_PREFIX + (this.widgets?.length ?? 0),
					["STRING", { multiline: true }],
					app
				).widget;
				w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.6;
				w.value = String(value);
				w.serialize = false;
			}

			requestAnimationFrame(() => {
				const sz = this.computeSize();
				if (sz[0] < this.size[0]) {
					sz[0] = this.size[0];
				}
				if (sz[1] < this.size[1]) {
					sz[1] = this.size[1];
				}
				this.onResize?.(sz);
				app.graph.setDirtyCanvas(true, false);
			});
		}

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			if (message?.text) {
				populate.call(this, message.text);
			}
		};
	},
});
