// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { cloneDeep, range } from 'lodash';

import {
    ActionParameterType,
    BaseCollectionAction,
    Job,
    ObjectState,
    ObjectType,
    ShapeType,
    Source,
    Task,
} from 'cvat-core-wrapper';

type Collection = Parameters<BaseCollectionAction['run']>[0]['collection'];
type Track = Collection['tracks'][0];
type Shape = Collection['shapes'][0];

type Sam3ShapeType = 'polygon' | 'rectangle';

function toNumber(value: string | number | undefined, defaultValue: number): number {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : defaultValue;
}

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
    // This is intentionally local-storage based to avoid CVAT backend changes for an MVP.
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

function rectangleToPolygonPoints(rectPoints: number[]): number[] {
    if (rectPoints.length !== 4) {
        throw new Error('Expected rectangle points [xtl, ytl, xbr, ybr]');
    }
    const [xtl, ytl, xbr, ybr] = rectPoints;
    return [xtl, ytl, xbr, ytl, xbr, ybr, xtl, ybr];
}

function polygonToRectanglePoints(polyPoints: number[]): number[] {
    if (polyPoints.length < 6 || polyPoints.length % 2 !== 0) {
        throw new Error('Expected polygon points [x1,y1,x2,y2,...]');
    }
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (let i = 0; i < polyPoints.length; i += 2) {
        const x = polyPoints[i];
        const y = polyPoints[i + 1];
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
    }
    return [minX, minY, maxX, maxY].map((v) => Math.round(v));
}

type Sam3TrackRequest = {
    frames: string[];
    frame_indices: number[];
    conditioning: Array<{
        id: string;
        frame_index: number;
        shape_type: Sam3ShapeType;
        points: number[];
    }>;
    params: Record<string, unknown>;
};

type Sam3TrackResponse = {
    tracks: Record<string, Array<{
        frame_index: number;
        shape_type: Sam3ShapeType;
        points: number[];
        score: number;
    }>>;
};

export default class Sam3TrackAction extends BaseCollectionAction {
    #instance: Job | Task | null;
    #targetFrame: number;
    #outputShape: Sam3ShapeType;
    #overwriteAutoTracks: boolean;
    #convertShapesToTracks: boolean;
    #maxFramesPerRequest: number;
    #maxObjectsPerRequest: number;
    #jpegQuality: number;
    #sam2Model: string;

    public constructor() {
        super();
        this.#instance = null;
        this.#targetFrame = 0;
        this.#outputShape = 'polygon';
        this.#overwriteAutoTracks = true;
        this.#convertShapesToTracks = true;
        this.#maxFramesPerRequest = 24;
        this.#maxObjectsPerRequest = 4;
        this.#jpegQuality = 0.9;
        this.#sam2Model = 'sam2.1_b.pt';
    }

    public async init(instance: Job | Task, parameters: Record<string, string>): Promise<void> {
        this.#instance = instance;
        this.#targetFrame = +parameters['Target frame'];
        this.#outputShape = (parameters['Output shape'] as Sam3ShapeType) || 'polygon';
        this.#overwriteAutoTracks = parameters['Overwrite auto tracks'] === 'true';
        this.#convertShapesToTracks = parameters['Convert shapes to tracks'] === 'true';
        this.#maxFramesPerRequest = Math.max(0, Math.round(toNumber(parameters['Max frames / request (0=all)'], 24)));
        this.#maxObjectsPerRequest = Math.max(0, Math.round(toNumber(parameters['Max objects / request (0=all)'], 4)));
        this.#jpegQuality = Math.max(0.2, Math.min(1, toNumber(parameters['JPEG quality'], 0.9)));
        this.#sam2Model = parameters['SAM2.1 model'] || 'sam2.1_b.pt';
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
        const targetFrame = this.#targetFrame;
        if (currentFrame === targetFrame) {
            return noChanges;
        }

        const supportedSeedTypes = new Set<string>([ShapeType.POLYGON, ShapeType.RECTANGLE]);
        const resolveSeedPoints = (seedType: string, seedPoints: number[]): number[] => {
            if (this.#outputShape === 'polygon') {
                if (seedType === ShapeType.POLYGON) return seedPoints;
                if (seedType === ShapeType.RECTANGLE) return rectangleToPolygonPoints(seedPoints);
            } else {
                if (seedType === ShapeType.RECTANGLE) return seedPoints;
                if (seedType === ShapeType.POLYGON) return polygonToRectanglePoints(seedPoints);
            }
            throw new Error(`Unsupported seed type ${seedType} for output ${this.#outputShape}`);
        };

        // Build the list of frames from current -> target, respecting CVAT frame numbering.
        const frameNumbers = instance instanceof Job ?
            await instance.frames.frameNumbers() :
            range(0, instance.size);
        const minFrame = Math.min(currentFrame, targetFrame);
        const maxFrame = Math.max(currentFrame, targetFrame);
        let orderedFrames = frameNumbers.filter((f) => f >= minFrame && f <= maxFrame).sort((a, b) => a - b);
        if (targetFrame < currentFrame) {
            orderedFrames = orderedFrames.sort((a, b) => b - a);
        }

        const startIndex = orderedFrames.indexOf(currentFrame);
        const endIndex = orderedFrames.indexOf(targetFrame);
        if (startIndex === -1 || endIndex === -1 || endIndex < startIndex) {
            throw new Error('Target frame is out of job range or not available in this job');
        }
        const clipFrames = orderedFrames.slice(startIndex, endIndex + 1);
        if (!clipFrames.length) {
            return noChanges;
        }

        // Prepare conditioning (seed) shapes.
        type SeedMeta = {
            label_id: number;
            group: number;
            attributes: any[];
            z_order: number;
            rotation: number;
            originalShape?: Shape;
            originalTrack?: Track;
        };

        const conditioning: Sam3TrackRequest['conditioning'] = [];
        const metaById: Record<string, SeedMeta> = {};
        const processedLabelIDs = new Set<number>();
        const originalTracksByClientID = new Map<number, Track>();
        const originalShapesByClientID = new Map<number, Shape>();

        for (const shape of collection.shapes) {
            if (!supportedSeedTypes.has(shape.type)) continue;
            if (shape.frame !== currentFrame) continue;

            const points = resolveSeedPoints(shape.type, [...(shape.points || [])]);
            const id = `shape:${shape.clientID}`;
            conditioning.push({
                id,
                frame_index: currentFrame,
                shape_type: this.#outputShape,
                points,
            });
            metaById[id] = {
                label_id: shape.label_id,
                group: shape.group ?? 0,
                attributes: cloneDeep(shape.attributes || []),
                z_order: shape.z_order ?? 0,
                rotation: shape.rotation ?? 0,
                originalShape: shape,
            };
            processedLabelIDs.add(shape.label_id);
            originalShapesByClientID.set(shape.clientID as number, shape);
        }

        for (const track of collection.tracks) {
            const shapesBefore = (track.shapes || [])
                .filter((s) => s.frame <= currentFrame)
                .sort((a, b) => a.frame - b.frame);

            if (!shapesBefore.length) continue;
            const lastShape = shapesBefore[shapesBefore.length - 1];
            if (lastShape.outside) continue;
            if (!supportedSeedTypes.has(lastShape.type)) continue;

            const points = resolveSeedPoints(lastShape.type, [...(lastShape.points || [])]);
            const id = `track:${track.clientID}`;
            conditioning.push({
                id,
                frame_index: currentFrame,
                shape_type: this.#outputShape,
                points,
            });
            metaById[id] = {
                label_id: track.label_id,
                group: track.group ?? 0,
                attributes: cloneDeep(lastShape.attributes || []),
                z_order: lastShape.z_order ?? 0,
                rotation: lastShape.rotation ?? 0,
                originalTrack: track,
            };
            processedLabelIDs.add(track.label_id);
            originalTracksByClientID.set(track.clientID as number, track);
        }

        if (!conditioning.length) {
            throw new Error('No polygon/rectangle seeds found on the current frame (check filters and shapes/tracks)');
        }

        // For memory stability: chunk frames and/or objects instead of sending the whole clip in one request.
        const maxFramesPerRequest = this.#maxFramesPerRequest <= 0 ?
            clipFrames.length :
            Math.max(2, this.#maxFramesPerRequest);
        const maxObjectsPerRequest = this.#maxObjectsPerRequest;

        const conditioningChunks: Sam3TrackRequest['conditioning'][] = [];
        if (maxObjectsPerRequest > 0 && conditioning.length > maxObjectsPerRequest) {
            for (let i = 0; i < conditioning.length; i += maxObjectsPerRequest) {
                conditioningChunks.push(conditioning.slice(i, i + maxObjectsPerRequest));
            }
        } else {
            conditioningChunks.push(conditioning);
        }

        type TrackItem = Sam3TrackResponse['tracks'][string][number];
        const resultsById = new Map<string, Map<number, TrackItem>>();
        for (const c of conditioning) {
            resultsById.set(c.id, new Map<number, TrackItem>());
        }

        // Helper: encode and send one chunk.
        const callTrackChunk = async (
            frames: number[],
            cond: Sam3TrackRequest['conditioning'],
            cachedFirstFrame: { frame: number; b64: string } | null,
        ): Promise<{ response: Sam3TrackResponse; newCache: { frame: number; b64: string } | null }> => {
            const framesB64: string[] = [];
            const frameIndices: number[] = [];

            for (let i = 0; i < frames.length; i++) {
                if (cancelled()) {
                    return { response: { tracks: {} }, newCache: cachedFirstFrame };
                }

                const frameNumber = frames[i];
                const reuse = cachedFirstFrame && cachedFirstFrame.frame === frameNumber;
                if (reuse) {
                    framesB64.push(cachedFirstFrame.b64);
                    frameIndices.push(frameNumber);
                    continue;
                }

                const frameData = await instance.frames.get(frameNumber);
                if (frameData.deleted) {
                    if (i === 0) {
                        throw new Error(`Seed frame ${frameNumber} is deleted, cannot run SAM3 tracking`);
                    }
                    continue;
                }
                const data = await frameData.data();
                const imageBitmap = data.imageData as ImageBitmap;
                framesB64.push(await imageBitmapToJpegB64(imageBitmap, this.#jpegQuality));
                frameIndices.push(frameNumber);
            }

            if (!framesB64.length) {
                return { response: { tracks: {} }, newCache: cachedFirstFrame };
            }

            onProgress('Calling sam3_aa', 45);
            if (cancelled()) return { response: { tracks: {} }, newCache: cachedFirstFrame };

            const controller = new AbortController();
            const cancelPoll = window.setInterval(() => {
                if (cancelled()) controller.abort();
            }, 200);

            try {
                const baseUrl = getSam3AAURL();
                const url = `${baseUrl}/v1/track_video_shapes`;
                const payload: Sam3TrackRequest = {
                    frames: framesB64,
                    frame_indices: frameIndices,
                    conditioning: cond,
                    params: {
                        sam2_model: this.#sam2Model,
                    },
                };

                const response = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: controller.signal,
                });

                if (!response.ok) {
                    const text = await response.text();
                    throw new Error(`sam3_aa request failed (${response.status}): ${text}`);
                }

                const responseData = await response.json() as Sam3TrackResponse;
                const lastFrame = frames[frames.length - 1];
                const lastB64 = framesB64[frameIndices.indexOf(lastFrame)];
                const newCache = (Number.isInteger(lastFrame) && lastB64) ? { frame: lastFrame, b64: lastB64 } : null;
                return { response: responseData, newCache };
            } catch (error: any) {
                if (error?.name === 'AbortError') {
                    return { response: { tracks: {} }, newCache: cachedFirstFrame };
                }
                throw error;
            } finally {
                window.clearInterval(cancelPoll);
            }
        };

        // Run tracking (possibly chunked).
        onProgress('Preparing frames', 5);
        if (cancelled()) return noChanges;

        const totalObjectChunks = conditioningChunks.length;
        const frameStep = Math.max(1, maxFramesPerRequest - 1); // overlap 1 frame to chain seeds
        const totalFrameChunks = Math.max(1, Math.ceil((clipFrames.length - 1) / frameStep));
        const totalChunks = totalObjectChunks * totalFrameChunks;
        let finishedChunks = 0;

        for (const objChunk of conditioningChunks) {
            // Seed points we will carry chunk-to-chunk for each object id.
            const seedPointsById = new Map<string, number[]>();
            for (const c of objChunk) {
                seedPointsById.set(c.id, [...c.points]);
            }

            let cache: { frame: number; b64: string } | null = null;

            // Process frames in sliding windows: [0..N], [N..2N-1], ...
            for (let start = 0; start < clipFrames.length - 1; start += frameStep) {
                if (cancelled()) return noChanges;

                const end = Math.min(start + maxFramesPerRequest, clipFrames.length);
                const frameChunk = clipFrames.slice(start, end);
                if (frameChunk.length < 2) break;

                const seedFrame = frameChunk[0];
                const cond = objChunk.map((c) => ({
                    id: c.id,
                    frame_index: seedFrame,
                    shape_type: this.#outputShape,
                    points: seedPointsById.get(c.id) || c.points,
                }));

                // Encode + call backend.
                onProgress('Preparing frames', Math.min(44, 5 + Math.floor((finishedChunks / Math.max(1, totalChunks)) * 39)));
                const { response, newCache } = await callTrackChunk(frameChunk, cond, cache);
                cache = newCache;

                const tracksResp = response.tracks || {};
                for (const [id, items] of Object.entries(tracksResp)) {
                    const perFrame = resultsById.get(id) || new Map<number, TrackItem>();
                    for (const it of items || []) {
                        perFrame.set(it.frame_index, it);
                    }
                    resultsById.set(id, perFrame);
                }

                // Update seeds for the next frame chunk.
                const lastFrame = frameChunk[frameChunk.length - 1];
                for (const id of seedPointsById.keys()) {
                    const items = tracksResp[id] || [];
                    const last = items.find((it) => it.frame_index === lastFrame);
                    if (last?.points?.length) {
                        seedPointsById.set(id, last.points);
                    }
                }

                finishedChunks += 1;
            }
        }

        const tracksResp: Record<string, TrackItem[]> = {};
        for (const [id, byFrame] of resultsById.entries()) {
            const items = Array.from(byFrame.values()).sort((a, b) => a.frame_index - b.frame_index);
            tracksResp[id] = items;
        }

        onProgress('Building tracks', 80);
        if (cancelled()) return noChanges;

        const createdTracks: Track[] = [];
        const tracksToDelete: Track[] = [];

        // Always delete explicitly selected tracks we are rewriting.
        for (const tr of originalTracksByClientID.values()) {
            tracksToDelete.push(tr);
        }

        // Optionally delete auto tracks for processed labels (visible in current selection/filter set).
        if (this.#overwriteAutoTracks) {
            for (const tr of collection.tracks) {
                if (!processedLabelIDs.has(tr.label_id)) continue;
                if (tr.source !== Source.AUTO) continue;
                if (!originalTracksByClientID.has(tr.clientID as number)) {
                    tracksToDelete.push(tr);
                }
            }
        }

        const seenDeleteTrackClientIDs = new Set<number>();
        const dedupedTracksToDelete = tracksToDelete.filter((tr) => {
            const id = tr.clientID as number;
            if (seenDeleteTrackClientIDs.has(id)) return false;
            seenDeleteTrackClientIDs.add(id);
            return true;
        });

        for (const [id, items] of Object.entries(tracksResp)) {
            const meta = metaById[id];
            if (!meta) continue;

            const shapes = items
                .map((it) => ({
                    type: it.shape_type,
                    frame: it.frame_index,
                    attributes: cloneDeep(meta.attributes || []),
                    occluded: false,
                    outside: false,
                    points: it.points,
                    rotation: meta.rotation ?? 0,
                    z_order: meta.z_order ?? 0,
                }))
                .sort((a, b) => a.frame - b.frame);

            if (!shapes.length) continue;

            createdTracks.push({
                source: Source.AUTO,
                attributes: [],
                elements: [],
                frame: shapes[0].frame,
                group: meta.group,
                label_id: meta.label_id,
                shapes,
            } as Track);
        }

        const shapesToDelete: Shape[] = [];
        if (this.#convertShapesToTracks) {
            for (const sh of originalShapesByClientID.values()) {
                shapesToDelete.push(sh);
            }
        }

        onProgress('Committing', 95);
        if (cancelled()) return noChanges;

        return {
            created: { shapes: [], tags: [], tracks: createdTracks },
            deleted: { shapes: shapesToDelete, tags: [], tracks: dedupedTracksToDelete },
        };
    }

    public applyFilter(
        input: Parameters<BaseCollectionAction['applyFilter']>[0],
    ): ReturnType<BaseCollectionAction['applyFilter']> {
        const { collection, frameData } = input;
        const supportedTypes = new Set<string>([ShapeType.POLYGON, ShapeType.RECTANGLE]);

        return {
            shapes: collection.shapes
                .filter((shape) => shape.frame === frameData.number && supportedTypes.has(shape.type)),
            tags: [],
            tracks: collection.tracks.filter((track) => {
                if (!track.shapes?.length) return false;
                if (!supportedTypes.has(track.shapes[0].type)) return false;

                const shapesBefore = track.shapes
                    .filter((shape) => shape.frame <= frameData.number)
                    .sort((a, b) => a.frame - b.frame);
                if (!shapesBefore.length) return false;
                return !shapesBefore[shapesBefore.length - 1].outside;
            }),
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        return (
            objectState.objectType === ObjectType.SHAPE ||
            objectState.objectType === ObjectType.TRACK
        ) && (
            objectState.shapeType === ShapeType.POLYGON ||
            objectState.shapeType === ShapeType.RECTANGLE
        );
    }

    public get name(): string {
        return 'SAM2.1 Track (Run action)';
    }

    public get parameters(): BaseCollectionAction['parameters'] {
        return {
            'SAM2.1 model': {
                type: ActionParameterType.SELECT,
                values: ['sam2.1_b.pt', 'sam2.1_s.pt', 'sam2.1_t.pt', 'sam2.1_l.pt'],
                defaultValue: 'sam2.1_b.pt',
            },
            'Output shape': {
                type: ActionParameterType.SELECT,
                values: ['polygon', 'rectangle'],
                defaultValue: 'polygon',
            },
            'Max frames / request (0=all)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '500', '1'],
                defaultValue: '24',
            },
            'Max objects / request (0=all)': {
                type: ActionParameterType.NUMBER,
                values: ['0', '200', '1'],
                defaultValue: '4',
            },
            'JPEG quality': {
                type: ActionParameterType.NUMBER,
                values: ['0.2', '1', '0.05'],
                defaultValue: '0.9',
            },
            'Convert shapes to tracks': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: String(this.#convertShapesToTracks),
            },
            'Overwrite auto tracks': {
                type: ActionParameterType.CHECKBOX,
                values: ['true', 'false'],
                defaultValue: String(this.#overwriteAutoTracks),
            },
            'Target frame': {
                type: ActionParameterType.NUMBER,
                values: ({ instance }) => {
                    if (instance instanceof Job) {
                        return [instance.startFrame, instance.stopFrame, 1].map((val) => val.toString());
                    }
                    return [0, instance.size - 1, 1].map((val) => val.toString());
                },
                defaultValue: ({ instance }) => {
                    if (instance instanceof Job) {
                        return instance.stopFrame.toString();
                    }
                    return (instance.size - 1).toString();
                },
            },
        };
    }
}
