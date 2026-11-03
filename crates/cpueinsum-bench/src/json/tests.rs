use super::*;

#[test]
fn rejects_non_json_numbers() {
    for text in [
        "[+2,3]", "[02,3]", "[.5e1,3]", "[1.,3]", "[1e,3]", "[1e+,3]",
    ] {
        assert!(parse(text).is_err(), "{text:?} must be rejected");
    }
    // The grammar still accepts what the corpus declares.
    assert_eq!(parse("[0,10,1.5e2,-3]").unwrap().as_arr().unwrap().len(), 4);
}

#[test]
fn rejects_unescaped_control_characters_in_strings() {
    assert!(parse("\"a\nb\"").is_err());
    assert!(parse("\"a\tb\"").is_err());
    assert_eq!(parse(r#""a\nb""#).unwrap().as_str(), Some("a\nb"));
}
