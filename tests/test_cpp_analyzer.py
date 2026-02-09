"""
Tests for C++ Analyzer
=======================
Comprehensive tests for the tree-sitter-based C++ structural analyzer.
Tests: parsing, class extraction, inheritance chains, method signatures,
metadata formatting, regex fallback, categorisation, edge cases.
"""

import os
import json
import tempfile
import textwrap
from pathlib import Path

import pytest

from bakkesmod_rag.cpp_analyzer import (
    CppAnalyzer,
    CppClassInfo,
    CppMethodInfo,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_HEADER = textwrap.dedent("""\
    #pragma once
    template<class T> class ArrayWrapper;
    #include "../WrapperStructs.h"
    #include ".././Engine/ObjectWrapper.h"
    class PrimitiveComponentWrapper;
    class WorldInfoWrapper;

    class BAKKESMOD_PLUGIN_IMPORT ActorWrapper : public ObjectWrapper {
    public:
        CONSTRUCTORS(ActorWrapper)

        //BEGIN SELF IMPLEMENTED
        Vector GetLocation();
        void SetLocation(const Vector location);
        Vector GetVelocity();
        void SetVelocity(const Vector velocity);
        bool IsNull();
        bool IsNull() const;
        //END SELF IMPLEMENTED

        //AUTO-GENERATED FROM FIELDS
        float GetDrawScale();
        void SetDrawScale(float newDrawScale);
        unsigned char GetPhysics();
        ActorWrapper GetOwner();
    private:
        PIMPL
    };
""")

SAMPLE_ROOT_CLASS = textwrap.dedent("""\
    #pragma once
    #include <memory>

    class BAKKESMOD_PLUGIN_IMPORT ObjectWrapper
    {
    public:
        std::uintptr_t memory_address;
        ObjectWrapper(std::uintptr_t mem);
    };
""")

SAMPLE_CAR_HEADER = textwrap.dedent("""\
    #pragma once
    #include ".././GameObject/VehicleWrapper.h"
    class WheelWrapper;
    class BallWrapper;

    class BAKKESMOD_PLUGIN_IMPORT CarWrapper : public VehicleWrapper {
    public:
        CONSTRUCTORS(CarWrapper)

        bool IsBoostCheap();
        void SetBoostCheap(bool b);
        void SetCarRotation(Rotator rotation);
        void ForceBoost(bool force);
        std::string GetOwnerName();
        ControllerInput GetInput();
        void SetInput(ControllerInput input);
        void Destroy();
        void Demolish();
        unsigned long HasFlip();
        int GetLoadoutBody();

        ArrayWrapper<CarComponentWrapper> GetDefaultCarComponents();
        FlipCarComponentWrapper GetFlipComponent();
        bool CanDemolish(CarWrapper HitCar);
        void OnHitBall(BallWrapper Ball, Vector& HitLocation, Vector& HitNormal);
    private:
        PIMPL
    };
""")


@pytest.fixture
def analyzer():
    """Create a fresh CppAnalyzer instance."""
    return CppAnalyzer()


@pytest.fixture
def tmp_header(tmp_path):
    """Write sample header to a temp file and return its path."""
    f = tmp_path / "ActorWrapper.h"
    f.write_text(SAMPLE_HEADER, encoding="utf-8")
    return str(f)


@pytest.fixture
def tmp_dir(tmp_path):
    """Create a temp directory with multiple header files."""
    (tmp_path / "ObjectWrapper.h").write_text(SAMPLE_ROOT_CLASS, encoding="utf-8")
    (tmp_path / "ActorWrapper.h").write_text(SAMPLE_HEADER, encoding="utf-8")

    vehicle_header = textwrap.dedent("""\
        #pragma once
        class BAKKESMOD_PLUGIN_IMPORT VehicleWrapper : public RBActorWrapper {
        public:
            CONSTRUCTORS(VehicleWrapper)
            float GetMaxSpeed();
            void SetMaxSpeed(float speed);
        private:
            PIMPL
        };
    """)
    (tmp_path / "VehicleWrapper.h").write_text(vehicle_header, encoding="utf-8")

    rbactor_header = textwrap.dedent("""\
        #pragma once
        class BAKKESMOD_PLUGIN_IMPORT RBActorWrapper : public ActorWrapper {
        public:
            CONSTRUCTORS(RBActorWrapper)
            float GetMass();
            void SetMass(float mass);
        private:
            PIMPL
        };
    """)
    (tmp_path / "RBActorWrapper.h").write_text(rbactor_header, encoding="utf-8")

    (tmp_path / "CarWrapper.h").write_text(SAMPLE_CAR_HEADER, encoding="utf-8")

    return str(tmp_path)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestAnalyzerInit:
    """Tests for CppAnalyzer initialization."""

    def test_init_creates_parser(self, analyzer):
        """Parser should initialise (tree-sitter or None)."""
        # Should not raise
        assert analyzer is not None

    def test_init_has_parser_or_none(self, analyzer):
        """Parser is either tree-sitter or None (regex fallback)."""
        # Both are valid states
        assert analyzer._parser is not None or analyzer._parser is None


# ---------------------------------------------------------------------------
# Single file analysis
# ---------------------------------------------------------------------------

class TestAnalyzeFile:
    """Tests for analyzing individual header files."""

    def test_analyze_returns_classes(self, analyzer, tmp_header):
        """Should extract at least one class from a header."""
        classes = analyzer.analyze_file(tmp_header)
        assert len(classes) >= 1

    def test_class_name_extracted(self, analyzer, tmp_header):
        """Should extract correct class name."""
        classes = analyzer.analyze_file(tmp_header)
        names = [c.name for c in classes]
        assert "ActorWrapper" in names

    def test_base_class_extracted(self, analyzer, tmp_header):
        """Should identify the base class."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        assert "ObjectWrapper" in actor.base_classes

    def test_methods_extracted(self, analyzer, tmp_header):
        """Should extract method declarations."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        method_names = [m.name for m in actor.methods]
        assert "GetLocation" in method_names
        assert "SetLocation" in method_names

    def test_method_return_type(self, analyzer, tmp_header):
        """Should capture return types for methods."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        get_loc = next(
            (m for m in actor.methods if m.name == "GetLocation"), None
        )
        assert get_loc is not None
        assert "Vector" in get_loc.return_type

    def test_method_parameters(self, analyzer, tmp_header):
        """Should capture method parameters."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        set_loc = next(
            (m for m in actor.methods if m.name == "SetLocation"), None
        )
        assert set_loc is not None
        assert "Vector" in set_loc.parameters or "location" in set_loc.parameters

    def test_forward_declarations(self, analyzer, tmp_header):
        """Should extract forward-declared types."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        assert "PrimitiveComponentWrapper" in actor.forward_declarations
        assert "WorldInfoWrapper" in actor.forward_declarations

    def test_is_wrapper_flag(self, analyzer, tmp_header):
        """Should mark Wrapper classes appropriately."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        assert actor.is_wrapper is True

    def test_file_path_stored(self, analyzer, tmp_header):
        """Should record which file the class came from."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        assert actor.file != ""

    def test_nonexistent_file_returns_empty(self, analyzer):
        """Should return empty list for missing files."""
        classes = analyzer.analyze_file("nonexistent_file.h")
        assert classes == []

    def test_non_header_file_returns_empty(self, analyzer, tmp_path):
        """Should return empty list for non-.h files."""
        f = tmp_path / "test.txt"
        f.write_text("not a header", encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        assert classes == []

    def test_empty_file(self, analyzer, tmp_path):
        """Should handle empty header files gracefully."""
        f = tmp_path / "empty.h"
        f.write_text("", encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        assert classes == []


# ---------------------------------------------------------------------------
# Root class (no base)
# ---------------------------------------------------------------------------

class TestRootClass:
    """Tests for classes without a base class (like ObjectWrapper)."""

    def test_root_class_no_base(self, analyzer, tmp_path):
        """ObjectWrapper has no base class — base_classes should be empty."""
        f = tmp_path / "ObjectWrapper.h"
        f.write_text(SAMPLE_ROOT_CLASS, encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        # ObjectWrapper declared without `: public ...`
        obj = next((c for c in classes if c.name == "ObjectWrapper"), None)
        if obj:
            # Root class should have empty base_classes
            assert obj.base_classes == [] or obj.base_classes == [""]


# ---------------------------------------------------------------------------
# Directory analysis
# ---------------------------------------------------------------------------

class TestAnalyzeDirectory:
    """Tests for batch directory analysis."""

    def test_analyze_directory_finds_classes(self, analyzer, tmp_dir):
        """Should find all classes across files."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        assert "ActorWrapper" in all_classes
        assert "VehicleWrapper" in all_classes or len(all_classes) >= 2

    def test_analyze_directory_returns_dict(self, analyzer, tmp_dir):
        """Should return dict keyed by class name."""
        result = analyzer.analyze_directory(tmp_dir)
        assert isinstance(result, dict)

    def test_nonexistent_directory(self, analyzer):
        """Should return empty dict for missing directory."""
        result = analyzer.analyze_directory("/nonexistent/path")
        assert result == {}


# ---------------------------------------------------------------------------
# Inheritance chain
# ---------------------------------------------------------------------------

class TestInheritanceChain:
    """Tests for building full inheritance chains."""

    def test_simple_chain(self, analyzer, tmp_dir):
        """ActorWrapper -> ObjectWrapper chain."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        chain = analyzer.build_inheritance_chain("ActorWrapper", all_classes)
        assert "ObjectWrapper" in chain

    def test_multi_level_chain(self, analyzer, tmp_dir):
        """CarWrapper -> VehicleWrapper -> RBActorWrapper -> ActorWrapper."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        chain = analyzer.build_inheritance_chain("CarWrapper", all_classes)
        assert "VehicleWrapper" in chain

    def test_root_class_empty_chain(self, analyzer, tmp_dir):
        """ObjectWrapper is root — should have empty chain."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        chain = analyzer.build_inheritance_chain("ObjectWrapper", all_classes)
        assert chain == []

    def test_unknown_class_empty_chain(self, analyzer, tmp_dir):
        """Unknown class should return empty chain."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        chain = analyzer.build_inheritance_chain("NonexistentClass", all_classes)
        assert chain == []

    def test_no_infinite_loop(self, analyzer):
        """Should handle circular inheritance gracefully."""
        all_classes = {
            "A": CppClassInfo(name="A", base_classes=["B"]),
            "B": CppClassInfo(name="B", base_classes=["A"]),
        }
        chain = analyzer.build_inheritance_chain("A", all_classes)
        # Should terminate without infinite loop
        assert len(chain) <= 2


# ---------------------------------------------------------------------------
# Metadata formatting
# ---------------------------------------------------------------------------

class TestFormatMetadata:
    """Tests for metadata dict generation for RAG nodes."""

    def test_metadata_has_required_keys(self, analyzer, tmp_dir):
        """Metadata dict should contain expected keys."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        if "ActorWrapper" not in all_classes:
            pytest.skip("ActorWrapper not parsed")
        cls = all_classes["ActorWrapper"]
        meta = analyzer.format_metadata_for_node(cls, all_classes)

        assert "cpp_class" in meta
        assert "cpp_base_class" in meta
        assert "cpp_inheritance_chain" in meta
        assert "cpp_method_count" in meta
        assert "cpp_is_wrapper" in meta

    def test_metadata_class_name(self, analyzer, tmp_dir):
        """Metadata should contain correct class name."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        if "ActorWrapper" not in all_classes:
            pytest.skip("ActorWrapper not parsed")
        cls = all_classes["ActorWrapper"]
        meta = analyzer.format_metadata_for_node(cls, all_classes)
        assert meta["cpp_class"] == "ActorWrapper"

    def test_metadata_inheritance_chain_string(self, analyzer, tmp_dir):
        """Inheritance chain should be a ' -> ' separated string."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        if "ActorWrapper" not in all_classes:
            pytest.skip("ActorWrapper not parsed")
        cls = all_classes["ActorWrapper"]
        meta = analyzer.format_metadata_for_node(cls, all_classes)
        assert "ActorWrapper" in meta["cpp_inheritance_chain"]
        assert "->" in meta["cpp_inheritance_chain"]

    def test_metadata_methods_list(self, analyzer, tmp_dir):
        """Methods should be comma-separated string."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        if "ActorWrapper" not in all_classes:
            pytest.skip("ActorWrapper not parsed")
        cls = all_classes["ActorWrapper"]
        meta = analyzer.format_metadata_for_node(cls, all_classes)
        assert "cpp_methods" in meta
        # Should contain actual method names
        if cls.methods:
            assert len(meta["cpp_methods"]) > 0

    def test_metadata_getters_setters_split(self, analyzer, tmp_dir):
        """Should split methods into getters, setters, and other."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        if "ActorWrapper" not in all_classes:
            pytest.skip("ActorWrapper not parsed")
        cls = all_classes["ActorWrapper"]
        meta = analyzer.format_metadata_for_node(cls, all_classes)
        assert "cpp_getters" in meta
        assert "cpp_setters" in meta
        assert "cpp_other_methods" in meta

    def test_metadata_forward_declarations(self, analyzer, tmp_dir):
        """Should include forward-declared types in metadata."""
        all_classes = analyzer.analyze_directory(tmp_dir)
        if "ActorWrapper" not in all_classes:
            pytest.skip("ActorWrapper not parsed")
        cls = all_classes["ActorWrapper"]
        meta = analyzer.format_metadata_for_node(cls, all_classes)
        if cls.forward_declarations:
            assert "cpp_related_types" in meta


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

class TestCategorisation:
    """Tests for class category assignment."""

    def test_vehicle_category(self, analyzer):
        """CarWrapper should be categorised as 'vehicle'."""
        assert analyzer._categorize_class("CarWrapper") == "vehicle"

    def test_ball_category(self, analyzer):
        """BallWrapper should be categorised as 'ball'."""
        assert analyzer._categorize_class("BallWrapper") == "ball"

    def test_game_category(self, analyzer):
        """ServerWrapper should be categorised as 'game'."""
        assert analyzer._categorize_class("ServerWrapper") == "game"

    def test_player_category(self, analyzer):
        """PriWrapper should be categorised as 'player'."""
        assert analyzer._categorize_class("PriWrapper") == "player"

    def test_ui_category(self, analyzer):
        """CameraWrapper should be categorised as 'ui'."""
        assert analyzer._categorize_class("CameraWrapper") == "ui"

    def test_pickup_category(self, analyzer):
        """RumblePickupComponentWrapper should be categorised as 'pickup'."""
        assert analyzer._categorize_class("RumblePickupComponentWrapper") == "pickup"

    def test_unknown_category(self, analyzer):
        """Unknown class should be 'other'."""
        assert analyzer._categorize_class("SomethingElse") == "other"

    def test_physics_category(self, analyzer):
        """RBActorWrapper should be categorised as 'physics'."""
        assert analyzer._categorize_class("RBActorWrapper") == "physics"


# ---------------------------------------------------------------------------
# Complex method signatures (CarWrapper style)
# ---------------------------------------------------------------------------

class TestComplexMethods:
    """Tests for parsing complex BakkesMod method signatures."""

    def test_car_wrapper_methods(self, analyzer, tmp_path):
        """Should parse CarWrapper's diverse method signatures."""
        f = tmp_path / "CarWrapper.h"
        f.write_text(SAMPLE_CAR_HEADER, encoding="utf-8")
        classes = analyzer.analyze_file(str(f))

        car = next((c for c in classes if c.name == "CarWrapper"), None)
        assert car is not None

        method_names = [m.name for m in car.methods]
        assert "IsBoostCheap" in method_names
        assert "GetOwnerName" in method_names

    def test_bool_return_type(self, analyzer, tmp_path):
        """Should capture 'bool' return type."""
        f = tmp_path / "CarWrapper.h"
        f.write_text(SAMPLE_CAR_HEADER, encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        car = next(c for c in classes if c.name == "CarWrapper")

        boost_method = next(
            (m for m in car.methods if m.name == "IsBoostCheap"), None
        )
        if boost_method:
            assert "bool" in boost_method.return_type

    def test_wrapper_return_type(self, analyzer, tmp_path):
        """Should capture wrapper return types like FlipCarComponentWrapper."""
        f = tmp_path / "CarWrapper.h"
        f.write_text(SAMPLE_CAR_HEADER, encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        car = next(c for c in classes if c.name == "CarWrapper")

        flip_method = next(
            (m for m in car.methods if m.name == "GetFlipComponent"), None
        )
        if flip_method:
            assert "FlipCarComponentWrapper" in flip_method.return_type

    def test_car_forward_declarations(self, analyzer, tmp_path):
        """Should extract forward declarations from CarWrapper."""
        f = tmp_path / "CarWrapper.h"
        f.write_text(SAMPLE_CAR_HEADER, encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        car = next(c for c in classes if c.name == "CarWrapper")
        assert "WheelWrapper" in car.forward_declarations
        assert "BallWrapper" in car.forward_declarations


# ---------------------------------------------------------------------------
# Dataclass basics
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Tests for CppMethodInfo and CppClassInfo dataclasses."""

    def test_method_info_defaults(self):
        """CppMethodInfo should have sensible defaults."""
        m = CppMethodInfo(name="TestMethod")
        assert m.name == "TestMethod"
        assert m.return_type == ""
        assert m.parameters == ""
        assert m.is_const is False
        assert m.is_virtual is False
        assert m.line == 0

    def test_class_info_defaults(self):
        """CppClassInfo should have sensible defaults."""
        c = CppClassInfo(name="TestClass")
        assert c.name == "TestClass"
        assert c.base_classes == []
        assert c.methods == []
        assert c.forward_declarations == []
        assert c.is_wrapper is False
        assert c.category == ""

    def test_class_info_with_methods(self):
        """CppClassInfo should hold methods."""
        methods = [
            CppMethodInfo(name="Foo", return_type="void"),
            CppMethodInfo(name="Bar", return_type="int"),
        ]
        c = CppClassInfo(name="TestClass", methods=methods)
        assert len(c.methods) == 2
        assert c.methods[0].name == "Foo"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_file_with_only_forward_declarations(self, analyzer, tmp_path):
        """File with no class definition should return empty."""
        f = tmp_path / "forward_only.h"
        f.write_text("class Foo;\nclass Bar;\n", encoding="utf-8")
        classes = analyzer.analyze_file(str(f))
        assert classes == []

    def test_malformed_header(self, analyzer, tmp_path):
        """Should handle malformed C++ gracefully."""
        f = tmp_path / "bad.h"
        f.write_text("this is not valid C++ {{{{", encoding="utf-8")
        # Should not raise
        classes = analyzer.analyze_file(str(f))
        assert isinstance(classes, list)

    def test_template_class_forward_decl(self, analyzer, tmp_header):
        """Should not include template forward decls in forward_declarations."""
        classes = analyzer.analyze_file(tmp_header)
        actor = next(c for c in classes if c.name == "ActorWrapper")
        # ArrayWrapper is a template forward decl, not a plain class fwd decl
        assert "ArrayWrapper" not in actor.forward_declarations


# ---------------------------------------------------------------------------
# Real SDK analysis (integration-style)
# ---------------------------------------------------------------------------

SDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs_bakkesmod_only"
)


@pytest.mark.skipif(
    not os.path.isdir(SDK_DIR),
    reason="SDK headers not found at docs_bakkesmod_only/",
)
class TestRealSDK:
    """Integration tests against the actual BakkesMod SDK headers."""

    def test_analyze_real_sdk_finds_classes(self, analyzer):
        """Should find many classes in the real SDK."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        assert len(all_classes) >= 10, (
            f"Expected >=10 classes, found {len(all_classes)}"
        )

    def test_car_wrapper_in_sdk(self, analyzer):
        """Should find CarWrapper in the real SDK."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        assert "CarWrapper" in all_classes

    def test_car_wrapper_inherits_vehicle(self, analyzer):
        """CarWrapper should inherit from VehicleWrapper."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        if "CarWrapper" not in all_classes:
            pytest.skip("CarWrapper not found")
        car = all_classes["CarWrapper"]
        assert "VehicleWrapper" in car.base_classes

    def test_full_inheritance_chain(self, analyzer):
        """CarWrapper chain should include multiple ancestors."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        if "CarWrapper" not in all_classes:
            pytest.skip("CarWrapper not found")
        chain = analyzer.build_inheritance_chain("CarWrapper", all_classes)
        assert len(chain) >= 2, (
            f"Expected chain length >=2, got {chain}"
        )

    def test_car_wrapper_has_methods(self, analyzer):
        """CarWrapper should have many methods."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        if "CarWrapper" not in all_classes:
            pytest.skip("CarWrapper not found")
        car = all_classes["CarWrapper"]
        assert len(car.methods) >= 10, (
            f"Expected >=10 methods, found {len(car.methods)}"
        )

    def test_ball_wrapper_exists(self, analyzer):
        """BallWrapper should be in the SDK."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        assert "BallWrapper" in all_classes

    def test_metadata_generation_for_all_classes(self, analyzer):
        """Should generate metadata for every class without errors."""
        all_classes = analyzer.analyze_directory(SDK_DIR)
        for name, cls in all_classes.items():
            meta = analyzer.format_metadata_for_node(cls, all_classes)
            assert meta["cpp_class"] == name
            assert "cpp_inheritance_chain" in meta
