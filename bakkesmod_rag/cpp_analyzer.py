"""
C++ Analyzer
=============
Extracts structural metadata from BakkesMod SDK C++ header files using
tree-sitter.  The extracted information — class names, inheritance chains,
method signatures, forward declarations — is injected into RAG index
node metadata so the LLM receives **C++ structural context** alongside
raw text.

This means when someone asks "What methods does CarWrapper have?", the
retrieval system can match on structured metadata, not just substring
similarity.

All analysis is **offline** (runs once at index time) and requires zero
external tools beyond ``tree-sitter`` + ``tree-sitter-cpp`` (both already
installed for hybrid chunking).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("bakkesmod_rag.cpp_analyzer")


@dataclass
class CppMethodInfo:
    """A single C++ method declaration.

    Attributes:
        name: Method name (e.g. ``"GetVelocity"``).
        return_type: Return type string (e.g. ``"Vector"``).
        parameters: Parameter list as a raw string.
        is_const: Whether the method is ``const``.
        is_virtual: Whether declared ``virtual``.
        is_override: Whether marked ``override``.
        line: Line number in source file.
    """

    name: str
    return_type: str = ""
    parameters: str = ""
    is_const: bool = False
    is_virtual: bool = False
    is_override: bool = False
    line: int = 0


@dataclass
class CppClassInfo:
    """Structural metadata for a single C++ class.

    Attributes:
        name: Class name (e.g. ``"CarWrapper"``).
        base_classes: Direct parent classes.
        methods: Public method declarations.
        forward_declarations: Forward-declared types in the file.
        file: Source file path.
        is_wrapper: Whether this looks like a BakkesMod wrapper class.
        category: High-level category (e.g. ``"vehicle"``, ``"pickup"``).
    """

    name: str
    base_classes: List[str] = field(default_factory=list)
    methods: List[CppMethodInfo] = field(default_factory=list)
    forward_declarations: List[str] = field(default_factory=list)
    file: str = ""
    is_wrapper: bool = False
    category: str = ""


class CppAnalyzer:
    """Extracts C++ structural metadata from header files using tree-sitter.

    Parses ``.h`` files to extract class hierarchies, method signatures,
    and forward declarations.  The results are used to enrich RAG index
    nodes with typed metadata.

    Usage::

        analyzer = CppAnalyzer()
        info = analyzer.analyze_file("docs_bakkesmod_only/CarWrapper.h")
        # info.classes[0].name == "CarWrapper"
        # info.classes[0].base_classes == ["VehicleWrapper"]
        # info.classes[0].methods[0].name == "IsBoostCheap"

    The analyzer also builds a full class hierarchy across all files::

        hierarchy = analyzer.analyze_directory("docs_bakkesmod_only")
        # hierarchy["CarWrapper"].base_classes == ["VehicleWrapper"]
        # hierarchy["BallWrapper"].base_classes == ["RBActorWrapper"]
    """

    def __init__(self) -> None:
        """Initialize the tree-sitter C++ parser."""
        self._parser = None
        self._language = None
        self._init_parser()

    def _init_parser(self) -> None:
        """Lazily initialise tree-sitter with the C++ grammar."""
        try:
            from tree_sitter import Parser
            import tree_sitter_cpp as tscpp

            self._language = tscpp.language()
            self._parser = Parser()
            self._parser.language = self._language
            logger.debug("Tree-sitter C++ parser initialised")
        except ImportError as e:
            logger.warning(
                "tree-sitter-cpp not available, falling back to regex: %s", e
            )
            self._parser = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_file(self, filepath: str) -> List[CppClassInfo]:
        """Analyze a single C++ header file.

        Args:
            filepath: Path to the ``.h`` file.

        Returns:
            List of :class:`CppClassInfo` found in the file.
        """
        path = Path(filepath)
        if not path.exists() or not path.suffix == ".h":
            return []

        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read %s: %s", filepath, e)
            return []

        if self._parser:
            return self._analyze_with_treesitter(source, str(path))
        else:
            return self._analyze_with_regex(source, str(path))

    def analyze_directory(self, directory: str) -> Dict[str, CppClassInfo]:
        """Analyze all ``.h`` files in a directory.

        Args:
            directory: Path to the directory.

        Returns:
            Dict mapping class name to :class:`CppClassInfo`.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return {}

        all_classes: Dict[str, CppClassInfo] = {}

        for h_file in sorted(dir_path.rglob("*.h")):
            classes = self.analyze_file(str(h_file))
            for cls in classes:
                all_classes[cls.name] = cls

        logger.info(
            "Analyzed %d classes from %s", len(all_classes), directory
        )
        return all_classes

    def build_inheritance_chain(
        self, class_name: str, all_classes: Dict[str, CppClassInfo]
    ) -> List[str]:
        """Build the full inheritance chain for a class.

        Args:
            class_name: Name of the class to trace.
            all_classes: Dict from :meth:`analyze_directory`.

        Returns:
            List of ancestor class names from immediate parent to root.
            E.g. ``["VehicleWrapper", "RBActorWrapper", "ActorWrapper",
            "ObjectWrapper"]`` for ``CarWrapper``.
        """
        chain: List[str] = []
        current = class_name
        visited: set = set()

        while current in all_classes and current not in visited:
            visited.add(current)
            info = all_classes[current]
            if info.base_classes:
                parent = info.base_classes[0]  # primary inheritance
                chain.append(parent)
                current = parent
            else:
                break

        return chain

    def format_metadata_for_node(
        self,
        cls: CppClassInfo,
        all_classes: Dict[str, CppClassInfo],
    ) -> Dict[str, str]:
        """Format class info as metadata dict for a RAG index node.

        The metadata is designed to be **searchable** by the retrieval
        system and **informative** for the LLM.

        Args:
            cls: Class info to format.
            all_classes: Full class hierarchy dict.

        Returns:
            Dict of metadata keys to inject into node metadata.
        """
        chain = self.build_inheritance_chain(cls.name, all_classes)

        method_names = [m.name for m in cls.methods]

        # Group methods by pattern
        getters = [m for m in method_names if m.startswith("Get")]
        setters = [m for m in method_names if m.startswith("Set")]
        other = [m for m in method_names if not m.startswith("Get") and not m.startswith("Set")]

        # Build method signatures (compact)
        signatures = []
        for m in cls.methods[:30]:  # Cap at 30 to avoid huge metadata
            sig = f"{m.return_type} {m.name}({m.parameters})"
            if m.is_const:
                sig += " const"
            signatures.append(sig.strip())

        metadata: Dict[str, str] = {
            "cpp_class": cls.name,
            "cpp_base_class": cls.base_classes[0] if cls.base_classes else "",
            "cpp_inheritance_chain": " -> ".join([cls.name] + chain),
            "cpp_method_count": str(len(cls.methods)),
            "cpp_methods": ", ".join(method_names[:30]),
            "cpp_getters": ", ".join(getters[:15]),
            "cpp_setters": ", ".join(setters[:15]),
            "cpp_other_methods": ", ".join(other[:15]),
            "cpp_is_wrapper": str(cls.is_wrapper),
            "cpp_category": cls.category,
        }

        if signatures:
            metadata["cpp_signatures"] = "; ".join(signatures[:20])

        if cls.forward_declarations:
            metadata["cpp_related_types"] = ", ".join(
                cls.forward_declarations[:15]
            )

        return metadata

    # ------------------------------------------------------------------
    # Tree-sitter analysis
    # ------------------------------------------------------------------

    def _analyze_with_treesitter(
        self, source: str, filepath: str
    ) -> List[CppClassInfo]:
        """Parse source with tree-sitter and extract class info."""
        tree = self._parser.parse(source.encode("utf-8"))
        root = tree.root_node
        classes: List[CppClassInfo] = []

        # Extract forward declarations
        fwd_decls = self._extract_forward_declarations(root, source)

        # Find class_specifier nodes
        for node in self._walk_nodes(root):
            if node.type == "class_specifier":
                cls = self._extract_class_from_node(node, source, filepath)
                if cls:
                    cls.forward_declarations = fwd_decls
                    cls.category = self._categorize_class(cls.name)
                    classes.append(cls)

        return classes

    def _extract_class_from_node(
        self, node, source: str, filepath: str
    ) -> Optional[CppClassInfo]:
        """Extract CppClassInfo from a tree-sitter class_specifier node."""
        # Get class name
        name_node = node.child_by_field_name("name")
        if not name_node:
            return None

        class_name = self._node_text(name_node, source)
        if not class_name or class_name in (
            "BAKKESMOD_PLUGIN_IMPORT", "T",
        ):
            return None

        # Get base classes
        base_classes = []
        for child in self._walk_nodes(node):
            if child.type == "base_class_clause":
                for base_child in self._walk_nodes(child):
                    if base_child.type == "type_identifier":
                        base_name = self._node_text(base_child, source)
                        if base_name and base_name != "public":
                            base_classes.append(base_name)

        # Get methods from the field_declaration_list (class body)
        methods: List[CppMethodInfo] = []
        body = node.child_by_field_name("body")
        if body:
            methods = self._extract_methods_from_body(body, source)

        is_wrapper = class_name.endswith("Wrapper") or "Pickup" in class_name

        return CppClassInfo(
            name=class_name,
            base_classes=base_classes,
            methods=methods,
            file=filepath,
            is_wrapper=is_wrapper,
        )

    def _extract_methods_from_body(
        self, body_node, source: str
    ) -> List[CppMethodInfo]:
        """Extract method declarations from a class body node."""
        methods: List[CppMethodInfo] = []

        for child in self._walk_nodes(body_node):
            if child.type == "function_definition":
                method = self._parse_method_node(child, source)
                if method:
                    methods.append(method)
            elif child.type == "declaration":
                method = self._parse_declaration_as_method(child, source)
                if method:
                    methods.append(method)
            elif child.type == "field_declaration":
                method = self._parse_declaration_as_method(child, source)
                if method:
                    methods.append(method)

        return methods

    def _parse_method_node(
        self, node, source: str
    ) -> Optional[CppMethodInfo]:
        """Parse a function_definition node into CppMethodInfo."""
        decl = node.child_by_field_name("declarator")
        if not decl:
            return None

        text = self._node_text(node, source)
        return self._parse_method_text(text, node.start_point[0] + 1)

    def _parse_declaration_as_method(
        self, node, source: str
    ) -> Optional[CppMethodInfo]:
        """Try to parse a declaration node as a method declaration."""
        text = self._node_text(node, source).strip()
        if "(" not in text or text.startswith("//"):
            return None
        # Skip macros like CONSTRUCTORS(...)
        if text.startswith("CONSTRUCTORS") or text.startswith("#"):
            return None

        return self._parse_method_text(text, node.start_point[0] + 1)

    def _parse_method_text(
        self, text: str, line: int
    ) -> Optional[CppMethodInfo]:
        """Parse a method declaration text into CppMethodInfo."""
        # Clean up the text
        text = text.strip().rstrip(";").strip()

        # Match: return_type name(params) [const] [override]
        match = re.match(
            r"(?:virtual\s+)?"
            r"([\w:<>&*\s,]+?)\s+"
            r"(\w+)\s*"
            r"\(([^)]*)\)"
            r"(\s*const)?"
            r"(\s*override)?",
            text,
        )
        if not match:
            return None

        return_type = match.group(1).strip()
        name = match.group(2).strip()
        params = match.group(3).strip()
        is_const = bool(match.group(4))
        is_override = bool(match.group(5))
        is_virtual = "virtual" in text.split(name)[0]

        # Skip constructors/destructors/macros
        if name in ("CONSTRUCTORS", "BAKKESMOD_PLUGIN", "public", "private", "protected"):
            return None

        return CppMethodInfo(
            name=name,
            return_type=return_type,
            parameters=params,
            is_const=is_const,
            is_virtual=is_virtual,
            is_override=is_override,
            line=line,
        )

    def _extract_forward_declarations(
        self, root, source: str
    ) -> List[str]:
        """Extract forward-declared class names."""
        fwd: List[str] = []
        for node in self._walk_nodes(root):
            if node.type == "declaration":
                text = self._node_text(node, source).strip()
                match = re.match(r"^class\s+(\w+)\s*;", text)
                if match:
                    name = match.group(1)
                    if name != "BAKKESMOD_PLUGIN_IMPORT":
                        fwd.append(name)
        return fwd

    @staticmethod
    def _walk_nodes(node):
        """Iterate over all child nodes (non-recursive single level)."""
        cursor = node.walk()
        if not cursor.goto_first_child():
            return
        yield cursor.node
        while cursor.goto_next_sibling():
            yield cursor.node

    @staticmethod
    def _node_text(node, source: str) -> str:
        """Get the text content of a tree-sitter node."""
        return source[node.start_byte:node.end_byte]

    # ------------------------------------------------------------------
    # Regex fallback (when tree-sitter not available)
    # ------------------------------------------------------------------

    def _analyze_with_regex(
        self, source: str, filepath: str
    ) -> List[CppClassInfo]:
        """Fallback: extract class info using regex patterns."""
        classes: List[CppClassInfo] = []

        # Find class declarations
        class_pattern = re.compile(
            r"class\s+(?:BAKKESMOD_PLUGIN_IMPORT\s+)?(\w+)"
            r"\s*:\s*public\s+(\w+)",
        )

        for match in class_pattern.finditer(source):
            class_name = match.group(1)
            base_class = match.group(2)

            # Find methods (simplified regex)
            methods = self._extract_methods_regex(source)

            # Forward declarations
            fwd = re.findall(r"^class\s+(\w+)\s*;", source, re.MULTILINE)
            fwd = [f for f in fwd if f != "BAKKESMOD_PLUGIN_IMPORT"]

            is_wrapper = class_name.endswith("Wrapper") or "Pickup" in class_name

            classes.append(CppClassInfo(
                name=class_name,
                base_classes=[base_class],
                methods=methods,
                forward_declarations=fwd,
                file=filepath,
                is_wrapper=is_wrapper,
                category=self._categorize_class(class_name),
            ))

        return classes

    def _extract_methods_regex(self, source: str) -> List[CppMethodInfo]:
        """Extract method declarations using regex (fallback)."""
        methods: List[CppMethodInfo] = []

        # Match lines like: ReturnType MethodName(params);
        method_re = re.compile(
            r"^\s+"
            r"(?:virtual\s+)?"
            r"([\w:<>&*]+(?:\s+[\w:<>&*]+)*)\s+"
            r"(\w+)\s*"
            r"\(([^)]*)\)\s*"
            r"(const\s*)?"
            r"(override\s*)?"
            r";",
            re.MULTILINE,
        )

        for i, match in enumerate(method_re.finditer(source)):
            name = match.group(2)
            if name in ("CONSTRUCTORS", "BAKKESMOD_PLUGIN"):
                continue
            methods.append(CppMethodInfo(
                name=name,
                return_type=match.group(1).strip(),
                parameters=match.group(3).strip(),
                is_const=bool(match.group(4)),
                is_override=bool(match.group(5)),
            ))

        return methods

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    # Category patterns for BakkesMod SDK classes
    _CATEGORY_MAP = {
        "vehicle": ["Car", "Vehicle", "Wheel", "Boost"],
        "ball": ["Ball"],
        "game": ["Server", "GameEvent", "Tutorial", "Replay", "GameEditor"],
        "player": ["Pri", "Player", "Controller", "Team"],
        "ui": ["HUD", "Hud", "Camera", "Spectator", "GuiManager", "Modal"],
        "physics": ["RBActor"],
        "item": ["Product", "Loadout", "Item", "OnlineProduct"],
        "pickup": ["Pickup", "Rumble"],
        "stat": ["Stat", "Sample", "Graph"],
        "data": ["Data", "Database", "Wrapper", "Save"],
    }

    @classmethod
    def _categorize_class(cls, class_name: str) -> str:
        """Assign a high-level category to a class name."""
        for category, keywords in cls._CATEGORY_MAP.items():
            if any(kw in class_name for kw in keywords):
                return category
        return "other"
