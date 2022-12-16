use std::error::Error;
use std::{env, fs};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::iter::zip;

use byteorder::{BigEndian, ByteOrder};

use nom::bytes::complete::{take, tag};
use nom::multi::{count, many0};
use nom::number::complete::{be_f64, be_u16, be_u32, be_u64, be_f32, be_i64};
use nom::combinator::map;
use nom::IResult;
use nom::branch::alt;
use nom::bits::bits;
use nom::bits::streaming::take as bits_take;
use bitvec::prelude::*;
use itertools::{Itertools, iproduct, multizip};

fn main() -> Result<(), Box<dyn std::error::Error>>{
  let mut input = buffer_read_file("202210200900")?;

  let nusd = parse_data(&mut input, parse_nusd)?;
  let cntl = parse_data(&mut input, parse_cntl);
  println!("{:?}", nusd);
  println!("{:?}", cntl);
  Ok(())
}

#[derive(Debug)]
struct NUSD{
  creator: String,
  file_size: u64,
  file_version: u32,
  record_size: u32,
  info_record_size: u32,
  subc_record_size: u32,
}

#[derive(Debug)]
struct CNTL{
  difinition_type: String,
  base_time_str: String,
  base_time: u32,
  time_unit: String,
  projection: String,
  x_size :u32,
  y_size :u32,
  basepoint_x :f32,
  basepoint_y :f32,
  basepoint_lat :f32,
  basepoint_lon :f32,
  distance_x :f32,
  distance_y :f32,
  standard_lat_1 :f32,
  standard_lon_1 :f32,
  standard_lat_2 :f32,
  standard_lon_2 :f32,
  other_lat_1 :f32,
  other_lon_1 :f32,
  other_lat_2 :f32,
  other_lon_2 :f32,
  value: String,
  members: Vec<String>,
  forecast_times_1: Vec<u32>,
  forecast_times_2: Vec<u32>,
  planes_1: Vec<String>,
  planes_2: Vec<String>,
  elements: Vec<String>,
}

impl CNTL{
  fn index(&self) -> Vec<(String,u32,String,String)>{
    iproduct!(
      self.members.iter(),
      self.forecast_times_1.iter(),
      self.planes_1.iter(),
      self.elements.iter()
    ).map(|(m,f,p,e)|(m.to_string(), *f,p.to_string(),e.to_string())).collect_vec()
  }
}

pub fn buffer_read_file(path: &str) -> Result<BufReader<File>, Box<dyn Error>>{
  Ok(BufReader::new(File::open(path)?))
}

pub fn read_to_buffer(input: &mut BufReader<File>, n: usize) -> Result<Box<[u8]>, Box<dyn Error>>{
  let mut buf = vec![0u8;n].into_boxed_slice();
  input.read_exact(&mut buf)?;
  Ok(buf)
}

fn parse_data<T, P>(input: &mut BufReader<File>, parser:P) -> Result<T, Box<dyn Error>>
where P: Fn(&[u8]) -> IResult<&[u8], T>{
  let block_size = BigEndian::read_u32(&*read_to_buffer(input, 4)?);
  let buf = &*read_to_buffer(input, block_size as usize)?;
  let (_,data) = parser(buf).map_err(|err| anyhow::format_err!("{:?}",err))?;
  input.consume(4);
  Ok(data)
}

fn take_string(count: usize) -> impl for <'a> Fn(&'a [u8])-> IResult<&'a [u8], String>{
  move |input: &[u8]|{
    let (input, str) = take(count)(input)?;
    let str = String::from_utf8(str.to_vec()).unwrap().trim().to_string();
    Ok((input, str))
  }
}

fn parse_header<'a>(header: &'a str) -> impl for <'b> Fn(&'b [u8])-> IResult<&'b [u8], &'b [u8]> + 'a{
  move |input: &[u8]|{
    let (input,name) = tag(header)(input)?;
    let (input,_) = be_u32(input)?;
    let (input,_) = take(4usize)(input)?;
    Ok((input,name))
  }
}

fn parse_nusd(input: &[u8]) -> IResult<&[u8], NUSD>{
  let (input, _) = parse_header("NUSD")(input)?;
  let (input,creator) = take_string(72usize)(input)?;
  let (input, file_size) = be_u64(input)?;
  let (input, file_version) = be_u32(input)?;
  let (input, _) = take(4usize)(input)?;
  let (input, record_size) = be_u32(input)?;
  let (input, info_record_size) = be_u32(input)?;
  let (input, subc_record_size) = be_u32(input)?;
  Ok((input, NUSD{
    creator,
    file_size,
    file_version,
    record_size,
    info_record_size,
    subc_record_size,
  }))
}

fn parse_cntl(input: &[u8]) -> IResult<&[u8], CNTL>{
  let (input, _) = parse_header("CNTL")(input)?;
  let (input, difinition_type) = take_string(16usize)(input)?;
  let (input, base_time_str) = take_string(12usize)(input)?;
  let (input, base_time) = be_u32(input)?;
  let (input, time_unit) = take_string(4usize)(input)?;
  let (input, member_size) = be_u32(input)?;
  let (input, forecast_time_size) = be_u32(input)?;
  let (input, plane_size) = be_u32(input)?;
  let (input, element_size) = be_u32(input)?;
  let (input, projection) = take_string(4usize)(input)?;
  let (input, x_size) = be_u32(input)?;
  let (input, y_size) = be_u32(input)?;
  let (input, basepoint_x) = be_f32(input)?;
  let (input, basepoint_y) = be_f32(input)?;
  let (input, basepoint_lat) = be_f32(input)?;
  let (input, basepoint_lon) = be_f32(input)?;
  let (input, distance_x) = be_f32(input)?;
  let (input, distance_y) = be_f32(input)?;
  let (input, standard_lat_1) = be_f32(input)?;
  let (input, standard_lon_1) = be_f32(input)?;
  let (input, standard_lat_2) = be_f32(input)?;
  let (input, standard_lon_2) = be_f32(input)?;
  let (input, other_lat_1) = be_f32(input)?;
  let (input, other_lon_1) = be_f32(input)?;
  let (input, other_lat_2) = be_f32(input)?;
  let (input, other_lon_2) = be_f32(input)?;
  let (input, value) = take_string(4usize)(input)?;
  let (input, _) = take((4 * (2 + 6)) as usize)(input)?;
  let (input, members) = count(take_string(4usize), member_size as usize)(input)?;
  let (input, forecast_times_1) = count(be_u32,  forecast_time_size as usize)(input)?;
  let (input, forecast_times_2) = count(be_u32,  forecast_time_size as usize)(input)?;
  let (input, planes_1) = count(take_string(6usize),  plane_size as usize)(input)?;
  let (input, planes_2) = count(take_string(6usize),  plane_size as usize)(input)?;
  let (input, elements) = count(take_string(6usize),  element_size as usize)(input)?;
  Ok((input, CNTL{
    difinition_type,
    base_time_str,
    base_time,
    time_unit,
    projection,
    x_size,
    y_size,
    basepoint_x,
    basepoint_y,
    basepoint_lat,
    basepoint_lon,
    distance_x,
    distance_y,
    standard_lat_1,
    standard_lon_1,
    standard_lat_2,
    standard_lon_2,
    other_lat_1,
    other_lon_1,
    other_lat_2,
    other_lon_2,
    value,
    members,
    forecast_times_1,
    forecast_times_2,
    planes_1,
    planes_2,
    elements,
  }))
}