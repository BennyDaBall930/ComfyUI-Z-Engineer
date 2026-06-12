import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Z-Engineer frontend extras:
// 1. Shows the enhanced prompt(s) on the node after execution (issue #11).
// 2. Lets each multiline text box be resized vertically on its own via the
//    native textarea grip, without resizing the whole node (issue #9).
//    Heights persist with the workflow; double-click the grip strip to reset.

const PREVIEW_NODES = new Set(["ZEngineerEnhance", "ZEngineer"]);
const PREVIEW_PREFIX = "preview_";
const HEIGHTS_PROP = "zengineer.box_heights";
const MIN_BOX_HEIGHT = 40;
const MAX_BOX_HEIGHT = 2000;

function storedHeights(node) {
	node.properties ??= {};
	node.properties[HEIGHTS_PROP] ??= {};
	return node.properties[HEIGHTS_PROP];
}

function makeBoxResizable(node, w) {
	const el = w.inputEl;
	if (w.__zResizable || !(el instanceof HTMLTextAreaElement)) {
		return;
	}
	w.__zResizable = true;
	el.style.resize = "vertical";

	// Pin this widget to its user-chosen height during layout. The layout
	// engine treats minHeight === maxHeight as a fixed slot and gives the
	// remaining node space to the unpinned boxes.
	const origLayoutSize = w.computeLayoutSize?.bind(w);
	w.computeLayoutSize = function (n) {
		const base = origLayoutSize
			? origLayoutSize(n)
			: { minHeight: 60, maxHeight: 1e6, minWidth: 20, maxWidth: 1e6 };
		const h = node.properties?.[HEIGHTS_PROP]?.[w.name];
		if (h > 0) {
			base.minHeight = base.maxHeight = h;
		}
		return base;
	};

	// Translate a native-grip drag into widget units (canvas zoom aware).
	el.addEventListener("mousedown", () => {
		const startEl = el.offsetHeight;
		const startWidget = w.computedHeight ?? startEl;
		const onUp = () => {
			window.removeEventListener("mouseup", onUp);
			requestAnimationFrame(() => {
				const delta = el.offsetHeight - startEl;
				if (Math.abs(delta) < 4) {
					return;
				}
				const scale = app.canvas?.ds?.scale || 1;
				const next = Math.round(startWidget + delta / scale);
				storedHeights(node)[w.name] = Math.min(MAX_BOX_HEIGHT, Math.max(MIN_BOX_HEIGHT, next));
				app.graph.setDirtyCanvas(true, true);
			});
		};
		window.addEventListener("mouseup", onUp);
	});

	// Double-click near the resize grip to clear the pin.
	el.addEventListener("dblclick", (ev) => {
		const rect = el.getBoundingClientRect();
		if (ev.clientY > rect.bottom - 14 && ev.clientX > rect.right - 24) {
			delete storedHeights(node)[w.name];
			app.graph.setDirtyCanvas(true, true);
		}
	});
}

function makeAllBoxesResizable(node) {
	for (const w of node.widgets ?? []) {
		if (w.inputEl) {
			makeBoxResizable(node, w);
		}
	}
}

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
			makeAllBoxesResizable(this);

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
	nodeCreated(node) {
		if (PREVIEW_NODES.has(node.comfyClass)) {
			makeAllBoxesResizable(node);
		}
	},
});
