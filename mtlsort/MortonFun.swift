//
//  MortonFun.swift
//  mtlsort
//
//  Created by Carl Wieland on 1/9/20.
//  Copyright © 2020 Datum Apps. All rights reserved.
//

import Foundation
import Metal


class MortonFun {
    typealias DataType = Vector2

    let device = MTLCopyAllDevices()[1]

    lazy var library: MTLLibrary = {
        device.makeDefaultLibrary()!
    }()

    let shift = 10
    lazy var size = 1<<shift

    lazy var data: [DataType ] = [DataType](repeating: Vector2(x: 0, y: 0), count: size)

    init() {

        let squareSize = 1<<(shift >> 1)
        for y in 0..<squareSize {
            for x in 0..<squareSize  {
                data[(y * squareSize) + x].x = RealType(x)
                data[(y * squareSize) + x].y = RealType(y)
            }
        }

    }

    lazy var minMaxPipelineState: MTLComputePipelineState = {
        let function = library.makeFunction(name: "minMax")!
        return try! device.makeComputePipelineState(function: function)
    }()

    lazy var mortonPipelineState: MTLComputePipelineState = {
        let function = library.makeFunction(name: "mortonNumber")!
        return try! device.makeComputePipelineState(function: function)
    }()

    lazy var keyValueBitonicState: MTLComputePipelineState = {
        let function = library.makeFunction(name: "pointKeyValueBitonic")!
        return try! device.makeComputePipelineState(function: function)
    }()


    lazy var dataBuffer: MTLBuffer = {
        return  device.makeBuffer(bytes: &data, length: MemoryLayout<DataType>.stride * data.count, options: [.storageModeManaged])!
    }()

    func doStuff() {

        let (min, max) = calculateMinMax()
        let buffer = calculateMortonNumber(min: min, max: max)
        sortWithKeys(keyBuffer: buffer)
        //           print("CPU Bounding box:(\(minX),\(minY)), (\(maxX),\(maxY))")

    }

    func calculateMinMax() -> (RealType, RealType) {

        let commandQueue = device.makeCommandQueue()!
        let threadgroupsPerGrid = MTLSize(width: (size * 2) / minMaxPipelineState.threadExecutionWidth, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: minMaxPipelineState.threadExecutionWidth, height: 1, depth: 1)

        var resultSize: [Int32] = [-1,-1]
        let resultBuf = device.makeBuffer(bytes:&resultSize, length: MemoryLayout<Int32>.stride * resultSize.count, options: [.storageModeManaged])!

        let commands = commandQueue.makeCommandBuffer()!

        let encoder = commands.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(minMaxPipelineState)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuf, offset: 0, index: 1)
        encoder.setThreadgroupMemoryLength(minMaxPipelineState.threadExecutionWidth * MemoryLayout<Int32>.stride, index: 0)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        if let blitEncoder = commands.makeBlitCommandEncoder() {
            blitEncoder.synchronize(resource: resultBuf)
            blitEncoder.endEncoding()
        }

        commands.commit()
        commands.waitUntilCompleted()

        let boundingResults = resultBuf.contents().assumingMemoryBound(to: Int32.self)
        let boundingDataBuffer = UnsafeMutableBufferPointer(start: boundingResults, count: 2)
        let boundingResultArray = Array(boundingDataBuffer)
        let min = Int(boundingResultArray[0])
        let max = Int(boundingResultArray[1])
        var pointIndex = min / 2
        let minVal = min % 2 == 0 ? data[pointIndex].x : data[pointIndex].y
        pointIndex = max / 2
        let maxVal = max % 2 == 0 ? data[pointIndex].x : data[pointIndex].y
        print("Calculated min:\(minVal) max:\(maxVal)")
        return (minVal, maxVal)
    }

    func calculateMortonNumber(min: RealType, max: RealType) -> MTLBuffer {
        let commandQueue = device.makeCommandQueue()!
        let threadgroupsPerGrid = MTLSize(width: size/mortonPipelineState.threadExecutionWidth, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: mortonPipelineState.threadExecutionWidth, height: 1, depth: 1)

        var resultSize: [Int32] = [Int32](repeating: 0, count: data.count)
        let resultBuf = device.makeBuffer(bytes:&resultSize, length: MemoryLayout<Int32>.stride * resultSize.count, options: [.storageModeManaged])!

        let commands = commandQueue.makeCommandBuffer()!

        var minVal = min, maxVal = max

        let encoder = commands.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(mortonPipelineState)
        encoder.setBuffer(dataBuffer, offset: 0, index: 0)
        encoder.setBuffer(resultBuf, offset: 0, index: 1)
        encoder.setBytes(&minVal, length: MemoryLayout<RealType>.size, index: 2)
        encoder.setBytes(&maxVal, length: MemoryLayout<RealType>.size, index: 3)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        if let blitEncoder = commands.makeBlitCommandEncoder() {
            blitEncoder.synchronize(resource: resultBuf)
            blitEncoder.endEncoding()
        }

        commands.commit()
        commands.waitUntilCompleted()
//        let results = Array(UnsafeMutableBufferPointer(start: resultBuf.contents().assumingMemoryBound(to: Int32.self), count: resultSize.count))

        return resultBuf
    }

    func sortWithKeys(keyBuffer: MTLBuffer) {
        let commandQueue = device.makeCommandQueue()!
        let threadgroupsPerGrid = MTLSize(width: size/keyValueBitonicState.threadExecutionWidth, height: 1, depth: 1)
        let threadsPerThreadgroup = MTLSize(width: keyValueBitonicState.threadExecutionWidth, height: 1, depth: 1)

        let commands = commandQueue.makeCommandBuffer()!

        let logn = shift

        for i in 0..<logn {
            for j in 0...i {
                var p = UInt32(i)
                var q = UInt32(j)

                let encoder = commands.makeComputeCommandEncoder()!
                encoder.setComputePipelineState(keyValueBitonicState)
                encoder.setBuffer(keyBuffer, offset: 0, index: 0)
                encoder.setBuffer(dataBuffer, offset: 0, index: 1)
                encoder.setBytes(&p, length: MemoryLayout<UInt32>.stride, index: 2)
                encoder.setBytes(&q, length: MemoryLayout<UInt32>.stride, index: 3)
                encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
                encoder.endEncoding()
            }
        }
        if let blitEncoder = commands.makeBlitCommandEncoder() {
            blitEncoder.synchronize(resource: dataBuffer)
            blitEncoder.endEncoding()
        }
        commands.commit()
        commands.waitUntilCompleted()
        let dataPointer = dataBuffer.contents().assumingMemoryBound(to: DataType.self)
        let dataBufferPointer = UnsafeMutableBufferPointer(start: dataPointer, count: data.count)
        let resultsArray = Array.init(dataBufferPointer)

        print("GPU results:\(resultsArray)")
    }
}
