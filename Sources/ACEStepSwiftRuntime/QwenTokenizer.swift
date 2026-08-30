import Foundation
import Hub
import MLX
import Tokenizers

struct QwenTokenBatch {
    let inputIds: MLXArray
    let attentionMask: MLXArray
}

final class QwenTokenizer {
    private let encodeFunction: (String) -> [Int]
    private let tokenizer: Tokenizer

    let padTokenId: Int
    let maxLength: Int

    init(
        padTokenId: Int,
        maxLength: Int,
        tokenizer: Tokenizer,
        encode: @escaping (String) -> [Int]
    ) {
        self.padTokenId = padTokenId
        self.maxLength = maxLength
        self.tokenizer = tokenizer
        self.encodeFunction = encode
    }

    static func load(
        from directory: URL,
        maxLengthOverride: Int? = nil
    ) throws -> QwenTokenizer {
        let tokenizerDirectory = resolveTokenizerDirectory(directory)
        let tokenizerConfigURL = tokenizerDirectory.appendingPathComponent("tokenizer_config.json")
        let tokenizerDataURL = tokenizerDirectory.appendingPathComponent("tokenizer.json")

        guard FileManager.default.fileExists(atPath: tokenizerDirectory.path) else {
            throw QwenTokenizerError.directoryNotFound(tokenizerDirectory)
        }
        guard FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            throw QwenTokenizerError.fileNotFound(tokenizerConfigURL)
        }

        let tokenizerConfig = try decodeConfig(fileURL: tokenizerConfigURL)
        let tokenizer: Tokenizer
        if FileManager.default.fileExists(atPath: tokenizerDataURL.path) {
            let tokenizerData = try decodeConfig(fileURL: tokenizerDataURL)
            tokenizer = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        } else {
            let vocabURL = tokenizerDirectory.appendingPathComponent("vocab.json")
            let mergesURL = tokenizerDirectory.appendingPathComponent("merges.txt")
            guard FileManager.default.fileExists(atPath: vocabURL.path),
                  FileManager.default.fileExists(atPath: mergesURL.path) else {
                throw QwenTokenizerError.fileNotFound(tokenizerDataURL)
            }
            let tokenizerData = try makeBPETokenizerData(
                vocabURL: vocabURL,
                mergesURL: mergesURL
            )
            tokenizer = try AutoTokenizer.from(
                tokenizerConfig: tokenizerConfig,
                tokenizerData: tokenizerData
            )
        }

        let padTokenNode = tokenizerConfig["pad_token"]
        let padToken = padTokenNode.string() ?? padTokenNode["content"].string()
        guard let padToken else {
            throw QwenTokenizerError.padTokenMissing
        }
        guard let padTokenID = tokenizer.convertTokenToId(padToken)
            ?? tokenizer.eosTokenId
            ?? tokenizer.bosTokenId else {
            throw QwenTokenizerError.padTokenNotInVocabulary(padToken)
        }

        let resolvedMaxLength = maxLengthOverride
            ?? tokenizerConfig["model_max_length"].integer(or: 131_072)
        return QwenTokenizer(
            padTokenId: padTokenID,
            maxLength: resolvedMaxLength,
            tokenizer: tokenizer
        ) { text in
            tokenizer.encode(text: text)
        }
    }

    func encodePlain(
        prompts: [String],
        maxLength: Int? = nil
    ) -> QwenTokenBatch {
        precondition(!prompts.isEmpty, "At least one prompt must be provided.")
        let targetLength = min(maxLength ?? self.maxLength, self.maxLength)
        precondition(targetLength > 0, "Maximum sequence length must be positive.")

        var inputSequences: [[Int]] = []
        var attentionSequences: [[Int]] = []
        inputSequences.reserveCapacity(prompts.count)
        attentionSequences.reserveCapacity(prompts.count)

        for prompt in prompts {
            let tokens = Self.trim(
                encodeFunction(prompt),
                maxLength: targetLength
            )
            let (inputIDs, attentionMask) = Self.prepareSequence(
                tokens: tokens,
                padTokenID: padTokenId,
                maxLength: targetLength
            )
            inputSequences.append(inputIDs)
            attentionSequences.append(attentionMask)
        }

        let inputIDs = MLXArray(
            inputSequences.flatMap { $0 }.map(Float32.init),
            [prompts.count, targetLength]
        ).asType(.int32)
        let attentionMask = MLXArray(
            attentionSequences.flatMap { $0 }.map(Float32.init),
            [prompts.count, targetLength]
        ).asType(.int32)
        return QwenTokenBatch(inputIds: inputIDs, attentionMask: attentionMask)
    }

    private static func decodeConfig(fileURL: URL) throws -> Config {
        try JSONDecoder().decode(Config.self, from: Data(contentsOf: fileURL))
    }

    private static func makeBPETokenizerData(
        vocabURL: URL,
        mergesURL: URL
    ) throws -> Config {
        let vocabData = try Data(contentsOf: vocabURL)
        guard let vocabObject = try JSONSerialization.jsonObject(
            with: vocabData
        ) as? [String: Any] else {
            throw QwenTokenizerError.fileNotFound(vocabURL)
        }

        var vocab: [String: Int] = [:]
        vocab.reserveCapacity(vocabObject.count)
        for (token, value) in vocabObject {
            if let id = value as? Int {
                vocab[token] = id
            }
        }

        let merges = try String(contentsOf: mergesURL, encoding: .utf8)
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }

        let tokenizerDictionary: [String: Any] = [
            "model": [
                "vocab": vocab,
                "merges": merges
            ],
            "preTokenizer": [
                "type": "ByteLevel",
                "addPrefixSpace": false,
                "trimOffsets": true,
                "useRegex": true
            ],
            "decoder": [
                "type": "ByteLevel"
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: tokenizerDictionary)
        return try JSONDecoder().decode(Config.self, from: data)
    }

    private static func trim(_ tokens: [Int], maxLength: Int) -> [Int] {
        Array(tokens.prefix(maxLength))
    }

    private static func prepareSequence(
        tokens: [Int],
        padTokenID: Int,
        maxLength: Int
    ) -> ([Int], [Int]) {
        let truncated = Array(tokens.prefix(maxLength))
        let paddingCount = max(0, maxLength - truncated.count)
        var padded = truncated
        padded.append(contentsOf: Array(repeating: padTokenID, count: paddingCount))
        var attention = Array(repeating: 1, count: truncated.count)
        attention.append(contentsOf: Array(repeating: 0, count: paddingCount))
        return (padded, attention)
    }

    private static func resolveTokenizerDirectory(_ directory: URL) -> URL {
        let tokenizerDirectory = directory.appendingPathComponent("tokenizer", isDirectory: true)
        return FileManager.default.fileExists(atPath: tokenizerDirectory.path)
            ? tokenizerDirectory
            : directory
    }
}

enum QwenTokenizerError: LocalizedError {
    case directoryNotFound(URL)
    case fileNotFound(URL)
    case padTokenMissing
    case padTokenNotInVocabulary(String)

    var errorDescription: String? {
        switch self {
        case let .directoryNotFound(url):
            "Qwen tokenizer 目錄不存在：\(url.path)"
        case let .fileNotFound(url):
            "Qwen tokenizer 檔案不存在：\(url.path)"
        case .padTokenMissing:
            "Qwen tokenizer 缺少 pad token。"
        case let .padTokenNotInVocabulary(token):
            "Qwen tokenizer 找不到 pad token：\(token)"
        }
    }
}
