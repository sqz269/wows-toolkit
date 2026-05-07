//! Constants for GameParams pickle dictionary keys.
//!
//! These replace the hardcoded `HashableValue::String("...".to_string().into())`
//! patterns scattered throughout `provider.rs` and `main.rs`.

// Ship top-level keys
pub const SHIP_UPGRADE_INFO: &str = "ShipUpgradeInfo";
pub const SHIP_ABILITIES: &str = "ShipAbilities";
pub const A_HULL: &str = "A_Hull";

// Upgrade dict keys
pub const UC_TYPE: &str = "ucType";
pub const COMPONENTS: &str = "components";

// ucType values
pub const UC_TYPE_HULL: &str = "_Hull";
pub const UC_TYPE_ARTILLERY: &str = "_Artillery";
pub const UC_TYPE_TORPEDOES: &str = "_Torpedoes";

// Component type keys (inside "components" dict).
//
// All known sub-keys WG uses across the corpus: 28 of them. The 13
// listed here are the ones that carry actual `HP_<id>.{model,armor,…}`
// hardpoints in any production / event ship. The other 15
// (`abilities` for non-mount ones, `engine`, `airSupport`,
// `flightControl`, `fighter`, `diveBomber`, `torpedoBomber`,
// `skipBomber`, `pinger`, `wcs`, `specials`, `innateSkills`,
// `axisLaser`, `chargeLasers`, `waves`) are gameplay-only —
// modifiers, plane definitions, weapon-control trees, water effects,
// etc. — and were verified to never expose `HP_*` subkeys in any
// Vehicle's component dict, so adding them here would be no-ops.
//
// `abilities` IS in this list because event ships (Amagi H2020,
// raider consumable variants) use `A_Abilities.HP_XGS_*` to anchor
// special-ability launchers. They have model paths and route to
// `accessories[]` via species=None.
pub const COMP_HULL: &str = "hull";
pub const COMP_ARTILLERY: &str = "artillery";
pub const COMP_ATBA: &str = "atba";
pub const COMP_AIR_DEFENSE: &str = "airDefense";
pub const COMP_AIR_ARMAMENT: &str = "airArmament";
pub const COMP_DIRECTORS: &str = "directors";
pub const COMP_FINDERS: &str = "finders";
pub const COMP_RADARS: &str = "radars";
pub const COMP_TORPEDOES: &str = "torpedoes";
pub const COMP_DEPTH_CHARGES: &str = "depthCharges";
pub const COMP_MISSILES: &str = "missiles";
pub const COMP_PHASER_LASERS: &str = "phaserLasers";
pub const COMP_ABILITIES: &str = "abilities";

/// Typed representation of component type keys.
///
/// `Ord` derives in declaration order (Hull → Artillery → Atba → AirDefense →
/// Directors → Finders → Radars → Torpedoes) so any `BTreeMap`
/// keyed by `ComponentType` iterates ships' mount families in a
/// fixed, semantically-grouped sequence — which gives placements-JSON
/// emission a stable order across runs regardless of `HashMap`
/// `RandomState`. See `mounts_by_type` in `types.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "rkyv", derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize))]
#[cfg_attr(feature = "rkyv", rkyv(derive(Hash, PartialEq, Eq, PartialOrd, Ord)))]
pub enum ComponentType {
    #[cfg_attr(feature = "serde", serde(rename = "hull"))]
    Hull,
    #[cfg_attr(feature = "serde", serde(rename = "artillery"))]
    Artillery,
    #[cfg_attr(feature = "serde", serde(rename = "atba"))]
    Atba,
    #[cfg_attr(feature = "serde", serde(rename = "airDefense"))]
    AirDefense,
    /// Aircraft armament: catapult mounts (HP_JC_*, HP_AC_*) carrying
    /// `Vehicle.<Hull>_AirArmament.HP_*.model = .../<asset>.model`. The
    /// catapult itself has GameParams `typeinfo.species: None,
    /// type: 'Catapult'`, so it falls through to the placements-JSON
    /// `accessories[]` section just like decorative mounts.
    #[cfg_attr(feature = "serde", serde(rename = "airArmament"))]
    AirArmament,
    #[cfg_attr(feature = "serde", serde(rename = "directors"))]
    Directors,
    #[cfg_attr(feature = "serde", serde(rename = "finders"))]
    Finders,
    #[cfg_attr(feature = "serde", serde(rename = "radars"))]
    Radars,
    #[cfg_attr(feature = "serde", serde(rename = "torpedoes"))]
    Torpedoes,
    /// Depth charge throwers + roller racks
    /// (`Vehicle.<Hull>_DepthChargeGuns.HP_AGB_*`,
    /// `HP_AGT_*`, `HP_BGT_*`, etc.). DDs and CLs use these on the
    /// fantail. `typeinfo.species == "DCharge"` routes them to
    /// placements-JSON `accessories[]` (no dedicated typed section yet).
    #[cfg_attr(feature = "serde", serde(rename = "depthCharges"))]
    DepthCharges,
    /// Modern missile launchers (Mk 141 Harpoon canisters, etc.)
    /// — `Vehicle.<Hull>_Missiles.HP_AGR_*`. Carries
    /// `typeinfo.species == "MissileGun"`. Modern / sci-fi event ships
    /// (Aegir, USN guided-missile destroyers).
    #[cfg_attr(feature = "serde", serde(rename = "missiles"))]
    Missiles,
    /// Star Trek event ships (`PXSB017_France_Borg_V2.A1_Lasers`):
    /// phaser laser turrets that re-skin standard `HP_FGM_*` mounts
    /// with `typeinfo.species == "Main"` so they show up as turrets.
    #[cfg_attr(feature = "serde", serde(rename = "phaserLasers"))]
    PhaserLasers,
    /// Event-ship special-ability HPs (`A_Abilities.HP_XGS_*` on the
    /// Amagi H2020 raider variant). Model is set but typeinfo carries
    /// `type=None / species=None`, so they land in `accessories[]`.
    /// Harmless to enumerate on production ships (no `HP_*` keys).
    #[cfg_attr(feature = "serde", serde(rename = "abilities"))]
    Abilities,
}

impl ComponentType {
    /// All known component types.
    pub const ALL: &[ComponentType] = &[
        Self::Hull,
        Self::Artillery,
        Self::Atba,
        Self::AirDefense,
        Self::AirArmament,
        Self::Directors,
        Self::Finders,
        Self::Radars,
        Self::Torpedoes,
        Self::DepthCharges,
        Self::Missiles,
        Self::PhaserLasers,
        Self::Abilities,
    ];

    /// The raw string key used in GameParams dictionaries.
    pub fn key(&self) -> &'static str {
        match self {
            Self::Hull => "hull",
            Self::Artillery => "artillery",
            Self::Atba => "atba",
            Self::AirDefense => "airDefense",
            Self::AirArmament => "airArmament",
            Self::Directors => "directors",
            Self::Finders => "finders",
            Self::Radars => "radars",
            Self::Torpedoes => "torpedoes",
            Self::DepthCharges => "depthCharges",
            Self::Missiles => "missiles",
            Self::PhaserLasers => "phaserLasers",
            Self::Abilities => "abilities",
        }
    }
}

impl std::fmt::Display for ComponentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hull => write!(f, "Hull"),
            Self::Artillery => write!(f, "Main Battery"),
            Self::Atba => write!(f, "Secondaries"),
            Self::AirDefense => write!(f, "AA"),
            Self::AirArmament => write!(f, "Air Armament"),
            Self::Directors => write!(f, "Directors"),
            Self::Finders => write!(f, "Finders"),
            Self::Radars => write!(f, "Radars"),
            Self::Torpedoes => write!(f, "Torpedoes"),
            Self::DepthCharges => write!(f, "Depth Charges"),
            Self::Missiles => write!(f, "Missiles"),
            Self::PhaserLasers => write!(f, "Phaser Lasers"),
            Self::Abilities => write!(f, "Abilities"),
        }
    }
}

/// All component type keys.
pub const ALL_COMPONENT_TYPES: &[&str] = &[
    COMP_HULL,
    COMP_ARTILLERY,
    COMP_ATBA,
    COMP_AIR_DEFENSE,
    COMP_AIR_ARMAMENT,
    COMP_DIRECTORS,
    COMP_FINDERS,
    COMP_RADARS,
    COMP_TORPEDOES,
    COMP_DEPTH_CHARGES,
    COMP_MISSILES,
    COMP_PHASER_LASERS,
    COMP_ABILITIES,
];

/// Component types that have 3D models (mounted on hull hardpoints).
pub const MODEL_COMPONENT_TYPES: &[&str] = &[
    COMP_HULL,
    COMP_ARTILLERY,
    COMP_ATBA,
    COMP_AIR_DEFENSE,
    COMP_AIR_ARMAMENT,
    COMP_DIRECTORS,
    COMP_FINDERS,
    COMP_RADARS,
    COMP_TORPEDOES,
    COMP_DEPTH_CHARGES,
    COMP_MISSILES,
    COMP_PHASER_LASERS,
    COMP_ABILITIES,
];

// Data field keys
pub const MODEL: &str = "model";
pub const ARMOR: &str = "armor";
pub const HIT_LOCATION_GROUPS: &str = "hitLocationGroups";
pub const HL_TYPE: &str = "hlType";
pub const MAX_HP: &str = "maxHP";
pub const REGENERATED_HP_PART: &str = "regeneratedHPPart";
pub const SPLASH_BOXES: &str = "splashBoxes";
pub const THICKNESS: &str = "thickness";
pub const DRAFT: &str = "draft";
pub const DOCK_Y_OFFSET: &str = "dockYOffset";
pub const VISIBILITY_FACTOR: &str = "visibilityFactor";
pub const VISIBILITY_FACTOR_BY_PLANE: &str = "visibilityFactorByPlane";
pub const MAX_DIST: &str = "maxDist";
pub const AMMO_LIST: &str = "ammoList";
pub const CAMOUFLAGE: &str = "camouflage";
pub const PERMOFLAGES: &str = "permoflages";
pub const TITLE: &str = "title";

// HP_ mount prefix
pub const HP_PREFIX: &str = "HP_";

// typeinfo keys
pub const TYPEINFO: &str = "typeinfo";
pub const TYPEINFO_TYPE: &str = "type";
pub const TYPEINFO_NATION: &str = "nation";
pub const TYPEINFO_SPECIES: &str = "species";

// Param identity keys
pub const PARAM_ID: &str = "id";
pub const PARAM_INDEX: &str = "index";
pub const PARAM_NAME: &str = "name";
