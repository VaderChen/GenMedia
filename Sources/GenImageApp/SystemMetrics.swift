import Darwin.Mach
import Foundation
import IOKit

struct SystemMetricsSnapshot: Encodable, Sendable {
    let ramUsedBytes: UInt64
    let ramTotalBytes: UInt64
    let ramUsage: Double?
    let gpuUsage: Double?
    let npuUsage: Double?

    static let unavailable = SystemMetricsSnapshot(
        ramUsedBytes: 0,
        ramTotalBytes: 0,
        ramUsage: nil,
        gpuUsage: nil,
        npuUsage: nil
    )
}

enum SystemMetricsReader {
    static func read() -> SystemMetricsSnapshot {
        let memory = readMemory()
        return SystemMetricsSnapshot(
            ramUsedBytes: memory.used,
            ramTotalBytes: memory.total,
            ramUsage: memory.ratio,
            gpuUsage: readGPUUsage(),
            npuUsage: readNPUUsage()
        )
    }

    private static func readMemory() -> (used: UInt64, total: UInt64, ratio: Double?) {
        let total = ProcessInfo.processInfo.physicalMemory
        var statistics = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        let result = withUnsafeMutablePointer(to: &statistics) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, rebound, &count)
            }
        }
        var pageSizeValue: vm_size_t = 0
        guard result == KERN_SUCCESS,
              host_page_size(mach_host_self(), &pageSizeValue) == KERN_SUCCESS,
              total > 0 else { return (0, total, nil) }

        let pageSize = UInt64(pageSizeValue)

        // Do not define usage as `total - free - speculative`: that also counts
        // reclaimable file cache and purgeable memory as pressure on RAM. This
        // makes the value noticeably higher than Activity Monitor and similar
        // tools. Use the resident app/anonymous pages, wired pages, and the
        // compressor pages instead, which matches their "memory used" view.
        let internalPages = UInt64(statistics.internal_page_count)
        let purgeablePages = min(internalPages, UInt64(statistics.purgeable_count))
        let appPages = internalPages - purgeablePages
        let usedPages = appPages
            + UInt64(statistics.wire_count)
            + UInt64(statistics.compressor_page_count)
        let used = min(total, usedPages * pageSize)
        return (used, total, min(1, max(0, Double(used) / Double(total))))
    }

    private static func readGPUUsage() -> Double? {
        guard let matching = IOServiceMatching("IOAccelerator") else { return nil }
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
            return nil
        }
        defer { IOObjectRelease(iterator) }

        var readings: [Double] = []
        var service = IOIteratorNext(iterator)
        while service != 0 {
            let currentService = service
            var properties: Unmanaged<CFMutableDictionary>?
            if IORegistryEntryCreateCFProperties(currentService, &properties, kCFAllocatorDefault, 0) == KERN_SUCCESS,
               let dictionary = properties?.takeRetainedValue() as? [String: Any],
               let statistics = dictionary["PerformanceStatistics"] as? [String: Any] {
                let value = statistics["Device Utilization %"]
                    ?? statistics["GPU Activity(%)"]
                    ?? statistics["Renderer Utilization %"]
                if let number = value as? NSNumber {
                    readings.append(min(1, max(0, number.doubleValue / 100)))
                }
            }
            IOObjectRelease(currentService)
            service = IOIteratorNext(iterator)
        }
        return readings.max()
    }

    private static func readNPUUsage() -> Double? {
        let serviceClasses = [
            "ANEDriver",
            "ANEHWDevice",
            "H1xANELoadBalancer",
            "H11ANEIn"
        ]
        let utilizationKeys = [
            "ANE Utilization %",
            "Neural Engine Utilization %",
            "Device Utilization %",
            "Utilization %",
            "Activity(%)"
        ]

        var readings: [Double] = []
        for serviceClass in serviceClasses {
            guard let matching = IOServiceMatching(serviceClass) else { continue }
            var iterator: io_iterator_t = 0
            guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
                continue
            }

            var service = IOIteratorNext(iterator)
            while service != 0 {
                let currentService = service
                var properties: Unmanaged<CFMutableDictionary>?
                if IORegistryEntryCreateCFProperties(currentService, &properties, kCFAllocatorDefault, 0) == KERN_SUCCESS,
                   let dictionary = properties?.takeRetainedValue() as? [String: Any],
                   let statistics = dictionary["PerformanceStatistics"] as? [String: Any] {
                    for key in utilizationKeys {
                        if let number = statistics[key] as? NSNumber {
                            readings.append(min(1, max(0, number.doubleValue / 100)))
                            break
                        }
                    }
                }
                IOObjectRelease(currentService)
                service = IOIteratorNext(iterator)
            }
            IOObjectRelease(iterator)
        }
        return readings.max()
    }
}
