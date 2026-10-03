import Testing
@testable import GenImageApp

struct AppUpdateCheckerTests {
    @Test func dateBuildTagsUseTheSameDisplayVersion() {
        #expect(AppUpdateChecker.normalizedVersion("v1.26.1003-build.2312") == "1.26.1003 build 2312")
        #expect(AppUpdateChecker.normalizedVersion(" 1.26.0103 build 0005 ") == "1.26.0103 build 0005")
        #expect(AppUpdateChecker.applicationVersion("1.26.1003", build: "2312") == "1.26.1003 build 2312")
        #expect(AppUpdateChecker.applicationVersion("1.26.1003 build 2312", build: "2359") == "1.26.1003 build 2312")
    }

    @Test func laterBuildsOnTheSameDateAreUpdates() {
        #expect(AppUpdateChecker.isNewer("1.26.1003 build 2313", than: "1.26.1003 build 2312"))
        #expect(!AppUpdateChecker.isNewer("1.26.1003 build 2312", than: "1.26.1003 build 2312"))
        #expect(!AppUpdateChecker.isNewer("1.26.1003 build 2311", than: "1.26.1003 build 2312"))
        #expect(AppUpdateChecker.isNewer("1.26.1003-build.2313", than: "1.26.1003 build 2312"))
    }

    @Test func dateChangesTakePriorityOverTheBuildTime() {
        #expect(AppUpdateChecker.isNewer("1.26.1004 build 0000", than: "1.26.1003 build 2359"))
        #expect(AppUpdateChecker.isNewer("1.27.0101 build 0000", than: "1.26.1231 build 2359"))
        #expect(!AppUpdateChecker.isNewer("1.26.1003 build 2359", than: "1.26.1004"))
    }

    @Test func legacyVersionsAndPrereleasesRemainComparable() {
        #expect(AppUpdateChecker.normalizedVersion("v1.26.1004") == "1.26.1004")
        #expect(AppUpdateChecker.normalizedVersion("v1.2.3+revision") == "1.2.3")
        #expect(AppUpdateChecker.applicationVersion("1.26.1004", build: nil) == "1.26.1004")
        #expect(AppUpdateChecker.applicationVersion("1.26.1004", build: "1.0") == "1.26.1004")
        #expect(AppUpdateChecker.isNewer("1.26.1003 build 0000", than: "1.26.1003"))
        #expect(AppUpdateChecker.isNewer("1.2.3", than: "1.2.3-beta.2"))
        #expect(!AppUpdateChecker.isNewer("1.2.3-beta.2", than: "1.2.3"))
    }

    @Test(arguments: ["v1.26.1003-build.2400", "v1.26.1003-build.2360", "v1.26.1003-build.123",
        "1.26.1003 build abcd", "1.26.1003-build.2312-build.2359", "", "latest"])
    func rejectsMalformedBuilds(_ version: String) {
        #expect(AppUpdateChecker.normalizedVersion(version) == nil)
    }
}
