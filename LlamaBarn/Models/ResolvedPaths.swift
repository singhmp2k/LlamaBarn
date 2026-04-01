import Foundation

/// Resolved file paths for a downloaded model.
/// Separates "what is this model" (CatalogEntry) from "where is it on disk".
struct ResolvedPaths {
  /// Absolute path to the main model file
  let modelFile: String
  /// Absolute paths to additional shard files (multi-part models)
  let additionalParts: [String]
  /// Absolute path to the mmproj file (vision models), nil if not applicable
  let mmprojFile: String?
  /// Whether this model is in the legacy flat directory (~/.llamabarn/)
  /// as opposed to the HF cache (~/.cache/huggingface/hub/)
  let isLegacy: Bool

  /// All file paths this model occupies on disk
  var allPaths: [String] {
    var paths = [modelFile]
    paths.append(contentsOf: additionalParts)
    if let mmproj = mmprojFile {
      paths.append(mmproj)
    }
    return paths
  }
}
