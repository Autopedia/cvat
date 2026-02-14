// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import {
    ActionParameterType,
    BaseCollectionAction,
    Job,
    ObjectState,
    Source,
    Task,
} from 'cvat-core-wrapper';

type Collection = Parameters<BaseCollectionAction['run']>[0]['collection'];
type Shape = Collection['shapes'][0];

type OutputShape = 'polygon' | 'rectangle';
type LabelScope = 'vehicle-part labels only' | 'all shape labels';
type ChainApprox = 'simple' | 'none' | 'tc89_l1' | 'tc89_kcos';

type PromptPayload = {
    id: number;
    name: string;
    prompt_text: string;
    prompt_variants: string[];
    priority: number;
    canonical_part: string | null;
};

type SegmentImageInstancesRequest = {
    image_b64: string;
    prompts: Array<{
        id: number;
        name: string;
        prompt_text: string;
        prompt_variants: string[];
        priority: number;
    }>;
    imgsz: number | null;
    conf_threshold: number;
    min_mask_area: number;
    max_instances_per_prompt: number;
    dedup_iou_threshold: number;
    grouping_dilate_px: number;
    polygon: {
        simplify_epsilon_ratio: number;
        simplify_epsilon_min: number;
        chain_approx: ChainApprox;
        densify_max_edge_length: number | null;
        max_points: number | null;
        min_area: number;
    };
    postprocess: {
        fill_holes: boolean;
        remove_small_components_min_area: number;
    };
};

type SegmentImageInstancesResponse = {
    results: Array<{
        instance_id: string;
        id: number | string;
        name: string;
        score: number;
        box_xyxy: number[];
        polygon_points: number[];
        vehicle_group: number;
    }>;
};

function normalizeBaseURL(url: string): string {
    return url.replace(/\/+$/, '');
}

function isLoopbackHostname(hostname: string): boolean {
    const h = (hostname || '').trim().toLowerCase();
    return h === 'localhost' || h === '127.0.0.1' || h === '0.0.0.0';
}

function isIPv4Hostname(hostname: string): boolean {
    const h = (hostname || '').trim();
    return /^\d{1,3}(\.\d{1,3}){3}$/.test(h);
}

function inferDefaultSam3AAURL(): string {
    const hostname = window.location.hostname || '127.0.0.1';
    const protocol = window.location.protocol === 'https:' ? 'https' : 'http';
    // By default, call sam3_aa on the same host as CVAT, but on the sam3_aa port.
    // (Users can still override via localStorage for custom setups.)
    return `${protocol}://${hostname}:8089`;
}

function getSam3AAURL(): string {
    // Example:
    //   localStorage.setItem('cvat.sam3_aa.url', 'http://<TAILSCALE_IP>:8089')
    const stored = window.localStorage.getItem('cvat.sam3_aa.url');
    const currentHostname = window.location.hostname || '';
    if (stored && stored.trim()) {
        try {
            const storedURL = new URL(stored);
            // If CVAT is accessed remotely, ignore stale overrides that break remote access:
            // - loopback (127.0.0.1/localhost)
            // - old fixed Tailscale IP when CVAT host IP changes
            const remoteCVAT = !isLoopbackHostname(currentHostname);
            const staleLoopback = remoteCVAT && isLoopbackHostname(storedURL.hostname);
            const staleOldIP = remoteCVAT &&
                isIPv4Hostname(currentHostname) &&
                isIPv4Hostname(storedURL.hostname) &&
                storedURL.hostname !== currentHostname;
            if (staleLoopback || staleOldIP) {
                window.localStorage.removeItem('cvat.sam3_aa.url');
                // eslint-disable-next-line no-console
                console.warn(`Ignoring and clearing localStorage['cvat.sam3_aa.url']=${stored} for remote CVAT host=${currentHostname}`);
            } else {
                return normalizeBaseURL(stored);
            }
        } catch (_err: any) {
            // Ignore invalid overrides.
        }
    }

    return normalizeBaseURL(inferDefaultSam3AAURL());
}

async function blobToB64(blob: Blob): Promise<string> {
    const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error('FileReader failed when encoding a frame'));
        reader.onload = () => {
            if (typeof reader.result === 'string') resolve(reader.result);
            else reject(new Error('Unexpected FileReader result type'));
        };
        reader.readAsDataURL(blob);
    });

    const comma = dataUrl.indexOf(',');
    return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

async function imageBitmapToJpegB64(imageBitmap: ImageBitmap, quality = 0.9): Promise<string> {
    const width = imageBitmap.width;
    const height = imageBitmap.height;

    const canvas: OffscreenCanvas | HTMLCanvasElement = typeof OffscreenCanvas !== 'undefined' ?
        new OffscreenCanvas(width, height) :
        (() => {
            const el = window.document.createElement('canvas');
            el.width = width;
            el.height = height;
            return el;
        })();

    const ctx = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null;
    if (!ctx) {
        throw new Error('Could not initialize 2D canvas context for frame encoding');
    }

    ctx.drawImage(imageBitmap, 0, 0);

    const blob = await new Promise<Blob>((resolve, reject) => {
        if ('convertToBlob' in canvas) {
            (canvas as OffscreenCanvas).convertToBlob({ type: 'image/jpeg', quality })
                .then(resolve)
                .catch(reject);
        } else {
            (canvas as HTMLCanvasElement).toBlob((b) => {
                if (b) resolve(b);
                else reject(new Error('Canvas.toBlob returned null'));
            }, 'image/jpeg', quality);
        }
    });

    return blobToB64(blob);
}

function normalizeLabelName(name: string): string {
    return name
        .toLowerCase()
        .replace(/[_-]+/g, ' ')
        .replace(/[^a-z0-9 ]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function dedupeStrings(values: string[]): string[] {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const v of values) {
        const t = v.trim();
        if (!t) continue;
        const k = t.toLowerCase();
        if (seen.has(k)) continue;
        seen.add(k);
        out.push(t);
    }
    return out;
}

const CANONICAL_ALIASES: Array<{ canonical: string; aliases: string[] }> = [
    { canonical: 'front_bumper', aliases: ['front bumper', 'vehicle front bumper'] },
    { canonical: 'rear_bumper', aliases: ['rear bumper', 'vehicle rear bumper'] },
    { canonical: 'front_fender', aliases: ['front fender', 'front quarter panel', 'front wheel arch'] },
    { canonical: 'rear_fender', aliases: ['rear fender', 'rear quarter panel', 'rear wheel arch'] },
    { canonical: 'side_step', aliases: ['side step', 'running board'] },
    { canonical: 'front_door', aliases: ['front door', 'front passenger door', 'front driver door'] },
    { canonical: 'rear_door', aliases: ['rear door', 'rear passenger door', 'rear driver door'] },
    { canonical: 'hood', aliases: ['hood', 'bonnet', 'front bonnet'] },
    { canonical: 'side_mirror', aliases: ['side mirror', 'wing mirror'] },
    { canonical: 'trunk', aliases: ['trunk', 'trunk lid', 'boot', 'boot lid'] },
    { canonical: 'headlight', aliases: ['headlight', 'head lamp', 'headlamp'] },
    { canonical: 'tail_light_quarter', aliases: ['tail light quarter', 'tail light outer', 'rear quarter tail light'] },
    { canonical: 'tail_light_trunk', aliases: ['tail light trunk', 'tail light inner', 'trunk tail light'] },
    { canonical: 'wheel', aliases: ['wheel', 'tire', 'tyre', 'rim', 'metal rim'] },
    { canonical: 'roof', aliases: ['roof'] },
];

const PROMPTS_BY_CANONICAL: Record<string, string[]> = {
    front_bumper: ['car front bumper', 'front bumper of a car'],
    rear_bumper: ['car rear bumper', 'rear bumper of a car'],
    front_fender: ['car front fender', 'front wheel arch', 'front quarter panel'],
    rear_fender: ['car rear fender', 'rear wheel arch', 'rear quarter panel'],
    side_step: ['car side step', 'running board'],
    front_door: ['car front door', 'front passenger door', 'front driver door'],
    rear_door: ['car rear door', 'rear passenger door', 'rear driver door'],
    hood: ['car hood', 'car bonnet'],
    side_mirror: ['car side mirror', 'wing mirror'],
    trunk: ['car trunk lid', 'car boot lid', 'car trunk'],
    headlight: ['car headlight', 'front headlamp'],
    tail_light_quarter: ['car tail light outer', 'car tail light on quarter panel'],
    tail_light_trunk: ['car tail light inner', 'car tail light on trunk lid'],
    wheel: ['car wheel', 'car tire', 'car rim', 'vehicle metal rim'],
    roof: ['car roof', 'vehicle roof panel'],
};

const PRIORITY_BY_CANONICAL: Record<string, number> = {
    wheel: 30,
    headlight: 25,
    tail_light_quarter: 25,
    tail_light_trunk: 25,
    side_mirror: 25,
    front_bumper: 15,
    rear_bumper: 15,
    hood: 12,
    trunk: 12,
    roof: 12,
    front_fender: 10,
    rear_fender: 10,
    side_step: 10,
    front_door: 8,
    rear_door: 8,
};

function inferCanonicalPart(labelName: string): string | null {
    const normalized = normalizeLabelName(labelName);
    if (!normalized) return null;

    for (const row of CANONICAL_ALIASES) {
        for (const alias of row.aliases) {
            if (normalized.includes(alias)) {
                return row.canonical;
            }
        }
    }
    return null;
}

function defaultPromptVariantsForLabel(labelName: string, canonical: string | null): string[] {
    const normalizedLabel = normalizeLabelName(labelName);
    const promptsFromCanonical = canonical ? (PROMPTS_BY_CANONICAL[canonical] || []) : [];

    const fallback = [
        normalizedLabel,
        `car ${normalizedLabel}`,
        `${normalizedLabel} of a car`,
    ];

    return dedupeStrings([...promptsFromCanonical, ...fallback]);
}

function canCreateShapesForLabelType(labelType: string): boolean {
    const t = String(labelType || '').toLowerCase();
    if (!t) return false;
    if (t === 'tag') return false;
    if (t === 'skeleton') return false;
    return true;
}

function shouldIncludeLabelByScope(labelName: string, labelScope: LabelScope): boolean {
    if (labelScope === 'all shape labels') {
        return true;
    }

    const canonical = inferCanonicalPart(labelName);
    if (canonical) {
        return true;
    }

    const normalized = normalizeLabelName(labelName);
    return normalized.includes('vehicle') || normalized.includes('car');
}

function toNumber(value: string | number | undefined, defaultValue: number): number {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : defaultValue;
}

function toBool(value: string | number | undefined, defaultValue: boolean): boolean {
    const s = String(value).trim().toLowerCase();
    if (s === 'true') return true;
    if (s === 'false') return false;
    return defaultValue;
}

function bboxToPolygon(boxXYXY: number[]): number[] {
    if (!Array.isArray(boxXYXY) || boxXYXY.length !== 4) return [];
    const xtl = Math.round(Number(boxXYXY[0]));
    const ytl = Math.round(Number(boxXYXY[1]));
    const xbr = Math.round(Number(boxXYXY[2]));
    const ybr = Math.round(Number(boxXYXY[3]));
    return [xtl, ytl, xbr, ytl, xbr, ybr, xtl, ybr];
}

function polygonToRectangle(polyPoints: number[]): number[] {
    if (polyPoints.length < 6 || polyPoints.length % 2 !== 0) return [];
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < polyPoints.length; i += 2) {
        const x = Number(polyPoints[i]);
        const y = Number(polyPoints[i + 1]);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return [];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
    }
    return [Math.round(minX), Math.round(minY), Math.round(maxX), Math.round(maxY)];
}

function buildPromptPayloads(
    labels: Array<{ id?: number; name: string; type: string; hasParent?: boolean }>,
    labelScope: LabelScope,
    usePromptVariants: boolean,
): PromptPayload[] {
    const prompts: PromptPayload[] = [];
    for (const label of labels) {
        if (!Number.isInteger(label.id)) continue;
        if (label.hasParent) continue;
        if (!canCreateShapesForLabelType(label.type)) continue;
        if (!shouldIncludeLabelByScope(label.name, labelScope)) continue;

        const canonical = inferCanonicalPart(label.name);
        const variants = defaultPromptVariantsForLabel(label.name, canonical);
        if (!variants.length) continue;

        const firstPrompt = variants[0];
        prompts.push({
            id: Number(label.id),
            name: label.name,
            prompt_text: firstPrompt,
            prompt_variants: usePromptVariants ? variants.slice(1) : [],
            priority: canonical ? (PRIORITY_BY_CANONICAL[canonical] ?? 0) : 0,
            canonical_part: canonical,
        });
    }
    return prompts;
}

export default class Sam3PartsZeroShotAction extends BaseCollectionAction {
    #instance: Job | Task | null;
    #outputShape: OutputShape;
    #labelScope: LabelScope;
    #usePromptVariants: boolean;
    #overwriteAutoShapes: boolean;
    #imgsz: number;
    #confThreshold: number;
    #minMaskArea: number;
    #maxInstancesPerLabel: number;
    #dedupIoUThreshold: number;
    #groupingDilatePx: number;
    #polySimplifyEpsRatio: number;
    #polySimplifyEpsMin: number;
    #polyMinArea: number;
    #polyMaxPoints: number;
    #polyDensifyMaxEdgeLength: number;
    #polyChainApprox: ChainApprox;
    #fillHoles: boolean;
    #removeSmallComponentsMinArea: number;

    public constructor() {
        super();
        this.#instance = null;
        this.#outputShape = 'polygon';
        this.#labelScope = 'vehicle-part labels only';
        this.#usePromptVariants = true;
        this.#overwriteAutoShapes = false;
        this.#imgsz = 1008;
        this.#confThreshold = 0.08;
        this.#minMaskArea = 500;
        this.#maxInstancesPerLabel = 0;
        this.#dedupIoUThreshold = 0.85;
        this.#groupingDilatePx = 24;
        this.#polySimplifyEpsRatio = 0.003;
        this.#polySimplifyEpsMin = 1.0;
        this.#polyMinArea = 16.0;
        this.#polyMaxPoints = 0;
        this.#polyDensifyMaxEdgeLength = 0;
        this.#polyChainApprox = 'simple';
        this.#fillHoles = false;
        this.#removeSmallComponentsMinArea = 0;
    }

    public async init(instance: Job | Task, parameters: Record<string, string | number>): Promise<void> {
        this.#instance = instance;
        this.#outputShape = String(parameters['Output shape'] || 'polygon') as OutputShape;
        this.#labelScope = String(parameters['Label scope'] || 'vehicle-part labels only') as LabelScope;
        this.#usePromptVariants = toBool(parameters['Use multi-prompt strategy'], true);
        this.#overwriteAutoShapes = toBool(parameters['Overwrite auto shapes on frame'], false);

        this.#imgsz = Math.max(128, Math.round(toNumber(parameters['SAM3 imgsz'], 1008)));
        this.#confThreshold = toNumber(parameters['Conf threshold'], 0.08);
        this.#minMaskArea = Math.max(0, Math.round(toNumber(parameters['Min mask area (px)'], 500)));
        this.#maxInstancesPerLabel = Math.max(0, Math.round(toNumber(parameters['Max instances / label (0=all)'], 0)));
        this.#dedupIoUThreshold = toNumber(parameters['Dedup IoU threshold'], 0.85);
        this.#groupingDilatePx = Math.max(0, Math.round(toNumber(parameters['Grouping dilate px'], 24)));

        this.#polySimplifyEpsRatio = toNumber(parameters['Polygon simplify eps ratio'], 0.003);
        this.#polySimplifyEpsMin = toNumber(parameters['Polygon simplify eps min'], 1.0);
        this.#polyMinArea = Math.max(0, toNumber(parameters['Polygon min area'], 16.0));
        this.#polyMaxPoints = Math.max(0, Math.round(toNumber(parameters['Polygon max points (0=none)'], 0)));
        this.#polyDensifyMaxEdgeLength = Math.max(0, toNumber(parameters['Polygon densify max edge (0=none)'], 0));
        this.#polyChainApprox = String(parameters['Polygon chain approx'] || 'simple') as ChainApprox;

        this.#fillHoles = toBool(parameters['Fill holes'], false);
        this.#removeSmallComponentsMinArea = Math.max(0, Math.round(
            toNumber(parameters['Remove small components area (px)'], 0),
        ));
    }

    public async destroy(): Promise<void> {
        // nothing to destroy
    }

    public async run(
        {
            collection,
            frameData: { number },
            onProgress,
            cancelled,
        }: Parameters<BaseCollectionAction['run']>[0],
    ): ReturnType<BaseCollectionAction['run']> {
        const noChanges = {
            created: { shapes: [], tags: [], tracks: [] },
            deleted: { shapes: [], tags: [], tracks: [] },
        };
        if (this.#instance === null) {
            return noChanges;
        }

        const instance = this.#instance;
        const currentFrame = number;

        onProgress('Preparing prompts', 8);
        const prompts = buildPromptPayloads(
            (instance.labels || []) as Array<{ id?: number; name: string; type: string; hasParent?: boolean }>,
            this.#labelScope,
            this.#usePromptVariants,
        );
        if (!prompts.length) {
            throw new Error(
                'No eligible labels for SAM3 parts segmentation. Check label types/scope and try again.',
            );
        }

        if (cancelled()) return noChanges;
        onProgress('Preparing frame', 22);

        const frame = await instance.frames.get(currentFrame);
        if (frame.deleted) {
            return noChanges;
        }

        const data = await frame.data();
        const imageBitmap = data.imageData as ImageBitmap;
        const frameB64 = await imageBitmapToJpegB64(imageBitmap);

        onProgress('Calling sam3_aa', 45);
        if (cancelled()) return noChanges;

        const payload: SegmentImageInstancesRequest = {
            image_b64: frameB64,
            prompts: prompts.map((prompt) => ({
                id: prompt.id,
                name: prompt.name,
                prompt_text: prompt.prompt_text,
                prompt_variants: prompt.prompt_variants,
                priority: prompt.priority,
            })),
            imgsz: Number.isFinite(this.#imgsz) ? this.#imgsz : null,
            conf_threshold: this.#confThreshold,
            min_mask_area: this.#minMaskArea,
            max_instances_per_prompt: this.#maxInstancesPerLabel,
            dedup_iou_threshold: this.#dedupIoUThreshold,
            grouping_dilate_px: this.#groupingDilatePx,
            polygon: {
                simplify_epsilon_ratio: this.#polySimplifyEpsRatio,
                simplify_epsilon_min: this.#polySimplifyEpsMin,
                chain_approx: this.#polyChainApprox,
                densify_max_edge_length: this.#polyDensifyMaxEdgeLength > 0 ? this.#polyDensifyMaxEdgeLength : null,
                max_points: this.#polyMaxPoints > 0 ? this.#polyMaxPoints : null,
                min_area: this.#polyMinArea,
            },
            postprocess: {
                fill_holes: this.#fillHoles,
                remove_small_components_min_area: this.#removeSmallComponentsMinArea,
            },
        };

        const controller = new AbortController();
        const cancelPoll = window.setInterval(() => {
            if (cancelled()) controller.abort();
        }, 200);

        let responseData: SegmentImageInstancesResponse;
        try {
            const baseUrl = getSam3AAURL();
            const response = await fetch(`${baseUrl}/v1/segment_image_instances`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: controller.signal,
            });

            if (!response.ok) {
                const text = await response.text();
                throw new Error(`sam3_aa request failed (${response.status}): ${text}`);
            }

            responseData = await response.json() as SegmentImageInstancesResponse;
        } catch (error: any) {
            if (error?.name === 'AbortError') {
                return noChanges;
            }
            throw error;
        } finally {
            window.clearInterval(cancelPoll);
        }

        if (cancelled()) return noChanges;
        onProgress('Building shapes', 78);

        const promptLabelIDs = new Set<number>(prompts.map((p) => p.id));
        const createdShapes: Shape[] = [];
        for (const [index, item] of (responseData.results || []).entries()) {
            const labelID = typeof item.id === 'number' ? item.id : Number(item.id);
            if (!Number.isInteger(labelID) || !promptLabelIDs.has(labelID)) {
                continue;
            }

            let points: number[] = [];
            if (this.#outputShape === 'polygon') {
                points = Array.isArray(item.polygon_points) ? item.polygon_points.map((v) => Math.round(Number(v))) : [];
                if (points.length < 6 || points.length % 2 !== 0) {
                    points = bboxToPolygon(item.box_xyxy || []);
                }
            } else {
                points = Array.isArray(item.box_xyxy) ? item.box_xyxy.map((v) => Math.round(Number(v))) : [];
                if (points.length !== 4) {
                    points = polygonToRectangle(item.polygon_points || []);
                }
            }
            if (!points.length) {
                continue;
            }

            createdShapes.push({
                type: this.#outputShape,
                frame: currentFrame,
                attributes: [],
                occluded: false,
                outside: false,
                points,
                rotation: 0,
                z_order: index,
                group: Number.isInteger(item.vehicle_group) ? item.vehicle_group : 0,
                label_id: labelID,
                source: Source.AUTO,
            } as Shape);
        }

        let deletedShapes: Shape[] = [];
        if (this.#overwriteAutoShapes) {
            deletedShapes = collection.shapes.filter((shape) => (
                shape.frame === currentFrame &&
                promptLabelIDs.has(shape.label_id) &&
                shape.source === Source.AUTO
            ));
        }

        if (cancelled()) return noChanges;
        onProgress('Committing', 95);

        return {
            created: { shapes: createdShapes, tags: [], tracks: [] },
            deleted: { shapes: deletedShapes, tags: [], tracks: [] },
        };
    }

    public applyFilter(
        input: Parameters<BaseCollectionAction['applyFilter']>[0],
    ): ReturnType<BaseCollectionAction['applyFilter']> {
        const { collection, frameData } = input;
        return {
            shapes: collection.shapes.filter((shape) => shape.frame === frameData.number),
            tags: [],
            tracks: [],
        };
    }

    public isApplicableForObject(_objectState: ObjectState): boolean {
        // This action can run without selecting an object, but keep it visible even when one is selected.
        return true;
    }

    public get name(): string {
        return 'SAM3 Parts Zero-shot (Run action)';
    }

    public get parameters(): BaseCollectionAction['parameters'] {
        return {
            'Output shape': {
                type: ActionParameterType.SELECT,
                values: ['polygon', 'rectangle'],
                defaultValue: 'polygon',
            },
            'Label scope': {
                type: ActionParameterType.SELECT,
                values: ['vehicle-part labels only', 'all shape labels'],
                defaultValue: 'vehicle-part labels only',
            },
            'Use multi-prompt strategy': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: 'true',
            },
            'Overwrite auto shapes on frame': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: 'false',
            },
            'SAM3 imgsz': {
                type: ActionParameterType.NUMBER,
                values: ['128', '4096', '8'],
                defaultValue: '1008',
            },
            'Conf threshold': {
                type: ActionParameterType.NUMBER,
                values: ['0', '1', '0.01'],
                defaultValue: '0.08',
            },
            'Min mask area (px)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '200000', '10'],
                defaultValue: '500',
            },
            'Max instances / label (0=all)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '100', '1'],
                defaultValue: '0',
            },
            'Dedup IoU threshold': {
                type: ActionParameterType.NUMBER,
                values: ['0.1', '1', '0.01'],
                defaultValue: '0.85',
            },
            'Grouping dilate px': {
                type: ActionParameterType.NUMBER,
                values: ['0', '300', '1'],
                defaultValue: '24',
            },
            'Polygon simplify eps ratio': {
                type: ActionParameterType.NUMBER,
                values: ['0', '0.1', '0.0005'],
                defaultValue: '0.003',
            },
            'Polygon simplify eps min': {
                type: ActionParameterType.NUMBER,
                values: ['0', '20', '0.1'],
                defaultValue: '1.0',
            },
            'Polygon min area': {
                type: ActionParameterType.NUMBER,
                values: ['0', '2000', '1'],
                defaultValue: '16',
            },
            'Polygon max points (0=none)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '5000', '1'],
                defaultValue: '0',
            },
            'Polygon densify max edge (0=none)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '100', '1'],
                defaultValue: '0',
            },
            'Polygon chain approx': {
                type: ActionParameterType.SELECT,
                values: ['simple', 'none', 'tc89_l1', 'tc89_kcos'],
                defaultValue: 'simple',
            },
            'Fill holes': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: 'false',
            },
            'Remove small components area (px)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '100000', '10'],
                defaultValue: '0',
            },
        };
    }
}
